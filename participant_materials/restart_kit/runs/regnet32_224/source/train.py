"""Fresh ImageNet fine-tuning. Calibration/audit predictions never select epochs."""
import argparse
import csv
import math
import os
from importlib.metadata import version
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from data import TrainImages, checked_rows, prepare_split, write_json
from model import build_model, head_parameters, image_transform, probability_from_logits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Numeric equivalence across GPUs/CPU is not guaranteed; CSV is generated on CPU.


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


@torch.inference_mode()
def collect_logits(net, loader, device, tta=False):
    net.eval()
    originals, flipped, labels = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=device.type == "cuda")
        originals.append(net(x).cpu())
        if tta:
            flipped.append(net(torch.flip(x, [3])).cpu())
        labels.append(y.cpu())
    return torch.cat(originals), torch.cat(labels), torch.cat(flipped) if tta else None


def metric(logits, labels, temperature=1.0, flips=None):
    # Competition clipping and float64 arithmetic, independent of training loss.
    p = probability_from_logits(logits.double(), temperature,
                                None if flips is None else flips.double()).numpy()
    p = np.clip(p, 1e-6, 1 - 1e-6)
    y = labels.numpy()
    loss = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    return float(loss.mean()), float(((p >= .5) == y).mean()), p, loss


def loader_for(root, rows, size, batch, workers=0, train=False, seed=42):
    return DataLoader(TrainImages(root, rows, image_transform(size, train)),
                      batch_size=batch, shuffle=train, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed))


def train(args):
    if args.epochs < 1 or not 0 <= args.warmup < args.epochs:
        raise ValueError("Need epochs > warmup >= 0")
    if args.batch < 2 or args.size < 64 or args.lr <= 0 or args.head_lr <= 0:
        raise ValueError("Invalid batch, size, or learning rate")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Use a NEW run folder to avoid overwriting: {args.out}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use the existing working GPU venv; do not reinstall blindly.")
    device = torch.device(args.device)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    set_seed(args.seed)
    rows = checked_rows(args.data, args.splits)
    train_rows = [r for r in rows if r["split"] == "train"]
    select_rows = [r for r in rows if r["split"] == "select"]
    tr_loader = loader_for(args.data, train_rows, args.size, args.batch, args.workers, True, args.seed)
    val_loader = loader_for(args.data, select_rows, args.size, args.batch, args.workers)
    net = build_model(args.arch, pretrained=not bool(args.init))
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=True), strict=True)
    args.out.mkdir(parents=True, exist_ok=True)
    # Starting weights also enable OFFLINE reproduction of training, within the ZIP size limit.
    torch.save(net.state_dict(), args.out / "initial_model.pt")
    shutil.copy2(args.splits, args.out / "splits.json")
    source = args.out / "source"
    source.mkdir()
    for name in ("model.py", "data.py", "train.py", "finalize.py", "predict.py", "make_package.py", "README_KO.md"):
        shutil.copy2(Path(__file__).parent / name, source / name)
    conf = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    conf.update(num_params=sum(p.numel() for p in net.parameters()),
                init_origin="ImageNet" if not args.init else str(args.init),
                counts={s: sum(r["split"] == s for r in rows)
                        for s in ("train", "select", "calibration", "audit")},
                python=platform.python_version(), torch=str(torch.__version__),
                torchvision=str(torchvision.__version__), cuda=torch.version.cuda,
                dependencies={name: version(name) for name in ("numpy", "pillow", "scipy", "scikit-learn")})
    write_json(args.out / "training_config.json", conf)
    net = net.to(device)
    head = list(head_parameters(net))
    head_ids = {id(p) for p in head}
    backbone = [p for p in net.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.lr}, {"params": head, "lr": args.head_lr}],
        weight_decay=1e-4)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss()  # No label smoothing in the initial controlled experiment.
    best, history = float("inf"), []
    print(f"device={device}; arch={args.arch}; params={conf['num_params']:,}", flush=True)
    print(f"roles={conf['counts']}; TTA={args.tta}; temperature=1.0", flush=True)
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        warming = epoch <= args.warmup
        for p in backbone:
            p.requires_grad_(not warming)
        net.train()
        if warming:
            # Frozen backbone includes BN buffers during the head-only warmup.
            net.eval()
            (net.fc if hasattr(net, "fc") else net.classifier).train()
        progress = max(0, epoch - args.warmup - 1) / max(1, args.epochs - args.warmup - 1)
        factor = .05 + .95 * (1 + math.cos(math.pi * progress)) / 2
        optimizer.param_groups[0]["lr"] = 0.0 if warming else args.lr * factor
        optimizer.param_groups[1]["lr"] = args.head_lr if warming else args.head_lr * factor
        total = 0.0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = criterion(net(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; no submission produced")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(net.parameters(), 5.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            total += loss.detach().item() * len(y)
        logits, labels, flips = collect_logits(net, val_loader, device, args.tta == "hflip")
        val_loss, acc, _, _ = metric(logits, labels, flips=flips)
        if not np.isfinite(val_loss):
            raise FloatingPointError("Non-finite validation probabilities")
        changed = val_loss < best
        if changed:
            best = val_loss
            state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            torch.save(state, args.out / "model.pt")
            write_json(args.out / "best.json", dict(epoch=epoch, select_logloss=val_loss, accuracy=acc))
        row = dict(epoch=epoch, train_loss=total / len(train_rows), select_logloss=val_loss,
                   select_accuracy=acc, backbone_lr=optimizer.param_groups[0]["lr"],
                   seconds=time.perf_counter() - start)
        history.append(row)
        with (args.out / "history.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(history)
        print(f"{epoch:02}/{args.epochs} train={row['train_loss']:.5f} "
              f"select={val_loss:.5f} acc={acc:.2%} "
              f"{row['seconds']:.1f}s {'BEST' if changed else ''}", flush=True)
    write_json(args.out / "completed.json", dict(epochs=args.epochs, best_select_logloss=best))
    print("Training complete. Calibration/audit subsets have not been used for fitting or epoch selection.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--data", type=Path, default=Path(".."))
    prep.add_argument("--out", type=Path, default=Path("splits.json"))
    prep.add_argument("--seed", type=int, default=20261107)
    tr = sub.add_parser("fit")
    tr.add_argument("--data", type=Path, default=Path(".."))
    tr.add_argument("--splits", type=Path, default=Path("splits.json"))
    tr.add_argument("--out", type=Path, required=True)
    tr.add_argument("--arch", choices=["regnet32", "regnet16", "b0"], default="regnet32")
    tr.add_argument("--size", type=int, default=224)
    tr.add_argument("--epochs", type=int, default=20)
    tr.add_argument("--warmup", type=int, default=1)
    tr.add_argument("--batch", type=int, default=16)
    tr.add_argument("--workers", type=int, default=0)
    tr.add_argument("--lr", type=float, default=5e-5)
    tr.add_argument("--head-lr", type=float, default=5e-4)
    tr.add_argument("--seed", type=int, default=20261107)
    tr.add_argument("--tta", choices=["none", "hflip"], default="none")
    tr.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    tr.add_argument("--init", type=Path, help="For offline reproduction: the supplied initial_model.pt")
    args = ap.parse_args()
    if args.command == "prepare":
        prepare_split(args.data, args.out, args.seed)
    else:
        train(args)


if __name__ == "__main__":
    main()

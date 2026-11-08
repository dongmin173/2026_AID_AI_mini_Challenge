"""224 RegNet training with raw/EMA selection; train-only BN recalibration."""
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
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn, update_bn

from data import TrainImages, checked_rows, prepare_split, write_json, read_json
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


def backward_step(loss, parameters, optimizer, scaler):
    """Never update weights using non-finite gradients; let optional FP16 recover.

    With FP16, scaled gradients can overflow even when the forward loss is finite.
    A non-finite norm must skip the update (including AdamW weight decay). Explicitly
    skipping also handles norm-reduction overflow when individual gradients are finite.
    FP32 keeps the strict non-finite check.
    """
    parameters = list(parameters)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = nn.utils.clip_grad_norm_(parameters, 5.0,
                                   error_if_nonfinite=not scaler.is_enabled())
    if not torch.isfinite(norm):
        if not scaler.is_enabled():
            raise FloatingPointError("Non-finite FP32 gradients; optimizer was not updated")
        scaler.update(new_scale=scaler.get_scale() * .5)
        optimizer.zero_grad(set_to_none=True)
        return False
    scaler.step(optimizer)
    scaler.update()
    return True


def train(args):
    if not 0 < args.ema_decay < 1:
        raise ValueError("EMA decay must be in (0,1)")
    if args.epochs < 1 or not 0 <= args.warmup < args.epochs:
        raise ValueError("Need epochs > warmup >= 0")
    if args.batch < 2 or args.size < 64 or args.lr <= 0 or args.head_lr <= 0:
        raise ValueError("Invalid batch, size, or learning rate")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Use a NEW run folder to avoid overwriting: {args.out}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use the existing working GPU venv; do not reinstall blindly.")
    if args.precision == "fp16" and args.device != "cuda":
        raise ValueError("--precision fp16 requires --device cuda")
    device = torch.device(args.device)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    set_seed(args.seed)
    rows = checked_rows(args.data, args.splits)
    train_rows = [r for r in rows if r["split"] == "train"]
    select_rows = [r for r in rows if r["split"] == "select"]
    tr_loader = loader_for(args.data, train_rows, args.size, args.batch, args.workers, True, args.seed)
    val_loader = loader_for(args.data, select_rows, args.size, args.batch, args.workers)
    # A separate deterministic loader; never use select/calibration/audit for BN fitting.
    bn_loader = DataLoader(TrainImages(args.data, train_rows, image_transform(args.size, train=False)),
                           batch_size=args.bn_batch, shuffle=True, num_workers=args.workers,
                           pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                           generator=torch.Generator().manual_seed(args.seed + 997))
    net = build_model(args.arch, pretrained=not bool(args.init))
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=True), strict=True)
    args.out.mkdir(parents=True, exist_ok=True)
    # Starting weights also enable OFFLINE reproduction of training, within the ZIP size limit.
    torch.save(net.state_dict(), args.out / "initial_model.pt")
    shutil.copy2(args.splits, args.out / "splits.json")
    source = args.out / "source"
    source.mkdir()
    for name in ("model.py", "data.py", "train.py", "finalize.py", "predict.py", "make_package.py", "README_KO.md", "train_ema.py", "finalize_ema.py"):
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
    amp = device.type == "cuda" and args.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0)
    criterion = nn.CrossEntropyLoss()  # No label smoothing in the initial controlled experiment.
    best, history = float("inf"), []
    ema = None
    ema_updates = 0
    variant_best = {"raw": float("inf"), "ema": float("inf")}
    variant_records = {}
    selected_variant = None
    print(f"device={device}; arch={args.arch}; params={conf['num_params']:,}; precision={args.precision}", flush=True)
    print(f"roles={conf['counts']}; TTA={args.tta}; temperature=1.0", flush=True)
    print(f"EMA decay={args.ema_decay}; starts after head warmup; train-only BN refresh each epoch", flush=True)
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
        total, seen, skipped, consecutive_skips = 0.0, 0, 0, 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = criterion(net(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; no submission produced")
            updated = backward_step(loss, net.parameters(), optimizer, scaler)
            if not updated:
                skipped += 1
                consecutive_skips += 1
                print(f"FP16 non-finite gradient norm: skipped update; scale={scaler.get_scale():g}", flush=True)
                if consecutive_skips >= 16:
                    raise FloatingPointError("Repeated FP16 failures; restart in a new run with --precision fp32")
                continue
            if not warming:
                if ema is None:
                    # First full-network update initializes EMA; warmup is not averaged.
                    ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay), use_buffers=False)
                ema.update_parameters(net)
                ema_updates += 1
            consecutive_skips = 0
            total += loss.detach().item() * len(y)
            seen += len(y)
        if seen == 0:
            raise FloatingPointError("No finite optimizer updates in this epoch; use --precision fp32")
        logits, labels, flips = collect_logits(net, val_loader, device, args.tta == "hflip")
        val_loss, acc, _, _ = metric(logits, labels, flips=flips)
        if not np.isfinite(val_loss):
            raise FloatingPointError("Non-finite validation probabilities")
        evaluations = [("raw", net, val_loss, acc)]
        ema_loss = ema_acc = None
        if ema is not None:
            # update_bn resets/estimates running stats using ONLY train images.
            # EMA weights are fixed during this forward-only pass.
            bn_loader.generator.manual_seed(args.seed + 997)
            update_bn(bn_loader, ema.module, device=device)
            ez, ey, ef = collect_logits(ema.module, val_loader, device, args.tta == "hflip")
            ema_loss, ema_acc, _, _ = metric(ez, ey, flips=ef)
            if not np.isfinite(ema_loss):
                raise FloatingPointError("Non-finite EMA validation loss")
            evaluations.append(("ema", ema.module, ema_loss, ema_acc))
        changed = False
        for variant, candidate, candidate_loss, candidate_acc in evaluations:
            if candidate_loss < variant_best[variant]:
                variant_best[variant] = candidate_loss
                variant_records[variant] = dict(epoch=epoch, select_logloss=candidate_loss, accuracy=candidate_acc)
                torch.save({k: v.detach().cpu() for k, v in candidate.state_dict().items()},
                           args.out / f"{variant}_best.pt")
            if candidate_loss < best:
                changed = True
                best = candidate_loss
                selected_variant = variant
                torch.save({k: v.detach().cpu() for k, v in candidate.state_dict().items()}, args.out / "model.pt")
                write_json(args.out / "best.json", dict(epoch=epoch, variant=variant,
                           select_logloss=candidate_loss, accuracy=candidate_acc))
        row = dict(epoch=epoch, train_loss=total / seen, select_logloss=val_loss,
                   select_accuracy=acc, ema_select_logloss=ema_loss, ema_select_accuracy=ema_acc,
                   backbone_lr=optimizer.param_groups[0]["lr"], seconds=time.perf_counter() - start,
                   updated_samples=seen, skipped_steps=skipped, grad_scale=scaler.get_scale(),
                   ema_updates=ema_updates)
        history.append(row)
        with (args.out / "history.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(history)
        ema_text = "warmup" if ema_loss is None else f"{ema_loss:.5f}"
        print(f"{epoch:02}/{args.epochs} train={row['train_loss']:.5f} "
              f"raw={val_loss:.5f} EMA={ema_text} best={best:.5f}({selected_variant}) "
              f"{row['seconds']:.1f}s skipped={skipped} {'BEST' if changed else ''}", flush=True)
    summary = dict(selected_variant=selected_variant, selected=read_json(args.out / "best.json"),
                   variant_best=variant_records, ema_decay=args.ema_decay,
                   bn_statistics_source="train split only; eval transform; fixed shuffled batches",
                   note="Only model.pt is deployed. raw_best.pt and ema_best.pt are training diagnostics, not an ensemble.")
    write_json(args.out / "selection_summary.json", summary)
    write_json(args.out / "source" / "selection_summary.json", summary)
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
    tr.add_argument("--precision", choices=["fp32", "fp16"], default="fp32",
                    help="Default fp32 avoids FP16 loss-scaling overflow; fp16 is optional")
    tr.add_argument("--ema-decay", type=float, default=0.99)
    tr.add_argument("--bn-batch", type=int, default=16)
    tr.add_argument("--init", type=Path, help="For offline reproduction: the supplied initial_model.pt")
    args = ap.parse_args()
    if args.command == "prepare":
        prepare_split(args.data, args.out, args.seed)
    else:
        train(args)


if __name__ == "__main__":
    main()

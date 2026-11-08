"""Train/fine-tune on train split only; select best model by raw validation CE."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from model import get_model, get_transform, parameter_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('..'), help='Folder with train/ test/')
    ap.add_argument('--arch', choices=('b0', 'b2'), required=True)
    ap.add_argument('--init', type=Path, help='Existing model state_dict (e.g. baseline_model.pt)')
    ap.add_argument('--pretrained', action='store_true', help='Download ImageNet weights during training only')
    ap.add_argument('--size', type=int, default=288)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=0, help='Windows-safe default')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if bool(args.init) == bool(args.pretrained):
        ap.error('Choose exactly one: --init (continue model) or --pretrained (new backbone)')
    if args.size < 128: ap.error('--size must be at least 128')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('CUDA not available: please fix GPU environment first')
    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_dir = args.data / 'train'
    train_all = ImageFolder(str(train_dir), transform=get_transform(args.size, train=True))
    val_all = ImageFolder(str(train_dir), transform=get_transform(args.size, train=False))
    assert train_all.class_to_idx == {'6e': 0, 'it': 1}, train_all.class_to_idx
    assert len(train_all) == 2906, f'Unexpected training image count: {len(train_all)}'
    idx = np.arange(len(train_all))
    y = np.array(train_all.targets)
    train_idx, val_idx = train_test_split(
        idx, test_size=0.20, random_state=args.seed, stratify=y)
    train_ds, val_ds = Subset(train_all, train_idx), Subset(val_all, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = get_model(args.arch, pretrained=args.pretrained)
    if args.init:
        initial = torch.load(args.init, map_location='cpu', weights_only=True)
        if isinstance(initial, dict) and 'state_dict' in initial:
            initial = initial['state_dict']
        model.load_state_dict(initial, strict=True)
    nparam = parameter_count(model)
    assert nparam <= 20_000_000
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda')
    criterion = torch.nn.CrossEntropyLoss()
    args.out.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    rows = []
    print(f'GPU={torch.cuda.get_device_name(0)} | {args.arch} | {nparam:,} parameters')
    print(f'Train={len(train_ds)} Val={len(val_ds)} | Size={args.size} | Batch={args.batch_size}')
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        train_total = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss.detach()) * labels.size(0)
        model.eval()
        val_total, n_correct = 0.0, 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)  # full precision validation
                val_total += torch.nn.functional.cross_entropy(
                    logits, labels, reduction='sum').item()
                n_correct += int((logits.argmax(1) == labels).sum().item())
        val_loss = val_total / len(val_ds)
        val_acc = n_correct / len(val_ds)
        scheduler.step(val_loss)
        changed = val_loss < best
        if changed:
            best = val_loss
            torch.save(model.state_dict(), args.out / 'best_model.pt')
        row = dict(epoch=epoch, train_loss=train_total / len(train_ds),
                   val_loss=val_loss, val_acc=val_acc,
                   lr=optimizer.param_groups[0]['lr'], seconds=time.time() - start)
        rows.append(row)
        with (args.out / 'history.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(rows)
        with (args.out / 'training_config.json').open('w', encoding='utf-8') as f:
            json.dump(dict(arch=args.arch, size=args.size, seed=args.seed,
                           num_params=nparam, best_val_loss=best,
                           epoch_saved=max(r['epoch'] for r in rows if r['val_loss'] == best),
                           train_count=len(train_ds), val_count=len(val_ds),
                           init_file=str(args.init) if args.init else 'ImageNet pretrained'),
                      f, indent=2, ensure_ascii=False)
        print(f'{epoch:02d}/{args.epochs} train={row["train_loss"]:.5f} '
              f'val_logloss={val_loss:.5f} val_acc={val_acc:.2%} '
              f'lr={row["lr"]:.2g} elapsed={row["seconds"]:.1f}s '
              f'{"BEST" if changed else ""}', flush=True)
    print(f'Best validation raw CE={best:.5f}; saved {args.out / "best_model.pt"}')


if __name__ == '__main__':
    main()

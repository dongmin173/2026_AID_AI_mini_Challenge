"""Offline, float32 CPU inference; the output is P(IT building = class 1)."""
import time
START_TIME = time.perf_counter()

import argparse
import csv
import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from model import build_model, image_transform, load_rgb, probability_from_logits


class TestImages:
    def __init__(self, root, size):
        self.files = sorted(p for p in Path(root).iterdir()
                            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if len(self.files) != 500 or len({p.stem for p in self.files}) != 500:
            raise ValueError("Need exactly 500 test images with unique filename stems")
        self.transform = image_transform(size, train=False)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p = self.files[i]
        return self.transform(load_rgb(p)), p.stem


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(test_dir, output, config, batch=16, threads=2, sample=None):
    torch.set_num_threads(max(1, min(threads, os.cpu_count() or 1)))
    conf = json.loads(Path(config).read_text(encoding="utf-8"))
    if conf["class_to_idx"] != {"6e": 0, "it": 1}:
        raise ValueError("Unexpected class mapping")
    for name, version in (("torch", torch.__version__), ("torchvision", torchvision.__version__)):
        if str(version).split("+")[0] != conf[name].split("+")[0]:
            warnings.warn(f"{name} differs from training ({conf[name]} vs {version}); verify CPU reproduction")
    weights = Path(config).resolve().parent / conf["checkpoint"]
    if file_hash(weights) != conf["model_sha256"]:
        raise ValueError("Weights changed after finalization")
    model = build_model(conf["arch"], pretrained=False)  # NEVER download weights in inference.
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.eval().cpu()
    count = sum(p.numel() for p in model.parameters())
    if count != conf["params"] or count > 20_000_000:
        raise ValueError(f"Wrong parameter count: {count}")
    if conf["tta"] not in ("none", "hflip") or not .1 <= conf["temperature"] <= 10:
        raise ValueError("Invalid inference options")
    dataset = TestImages(test_dir, conf["size"])
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0)
    ids, values = [], []
    with torch.inference_mode():
        for x, names in loader:
            logits = model(x).double()
            flips = model(torch.flip(x, [3])).double() if conf["tta"] == "hflip" else None
            p = probability_from_logits(logits, conf["temperature"], flips)
            ids.extend(names)
            values.extend(p.numpy().tolist())
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1 - 1e-6)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite prediction")
    if sample:
        with Path(sample).open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ["id", "prob"]:
                raise ValueError("Sample header must be id,prob")
            sample_ids = [r["id"] for r in reader]
        if len(sample_ids) != 500 or len(set(sample_ids)) != 500 or set(sample_ids) != set(ids):
            raise ValueError("Test IDs differ from sample_submission.csv")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prob"])
        writer.writerows(zip(ids, (format(float(v), ".10f") for v in values)))
    duration = time.perf_counter() - START_TIME
    if duration > 600:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"CPU inference exceeded 600 s ({duration:.1f}s)")
    tmp.replace(output)
    print(f"CPU elapsed (including imports): {duration:.1f}s / 600s; params={count:,}")
    print(f"CSV: {output.resolve()} -- 500 rows, P(IT=1)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", "--test_dir", dest="test_dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("submission.csv"))
    ap.add_argument("--config", type=Path, default=Path(__file__).parent / "inference_config.json")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--sample", type=Path)
    args = ap.parse_args()
    run(args.test_dir, args.output, args.config, args.batch, args.threads, args.sample)


if __name__ == "__main__":
    main()

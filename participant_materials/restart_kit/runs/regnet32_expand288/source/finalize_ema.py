"""Freeze the selected raw/EMA checkpoint and optionally calibrate; no audit evaluation."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from data import checked_rows, read_json, sha256_file, write_json
from model import build_model
from train import collect_logits, loader_for, metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path(".."))
    ap.add_argument("--calibrate", action="store_true", help="Optional; default temperature stays at 1")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    run = args.run
    if not (run / "completed.json").is_file():
        raise ValueError("Training did not complete")
    if (run / "inference_config.json").exists():
        raise FileExistsError("This run is already finalized. Keep the frozen configuration.")
    conf = read_json(run / "training_config.json")
    rows = checked_rows(args.data, run / "splits.json")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    net = build_model(conf["arch"], pretrained=False)
    net.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=True))
    net = net.to(device).eval()
    tta = conf["tta"] == "hflip"
    cal_rows = [r for r in rows if r["split"] == "calibration"]
    cal_loader = loader_for(args.data, cal_rows, conf["size"], args.batch)
    logits, labels, flips = collect_logits(net, cal_loader, device, tta)
    raw_cal = metric(logits, labels, flips=flips)[0]
    temperature = 1.0
    if args.calibrate:
        fit = minimize_scalar(lambda t: metric(logits, labels, t, flips)[0],
                              bounds=(0.75, 2.0), method="bounded", options={"xatol":1e-4})
        if fit.success and np.isfinite(fit.fun) and fit.fun < raw_cal:
            temperature = float(fit.x)
    cal_loss, cal_acc, _, _ = metric(logits, labels, temperature, flips)
    inf = dict(arch=conf["arch"], size=conf["size"], tta=conf["tta"], temperature=temperature,
               checkpoint="model.pt", model_sha256=sha256_file(run / "model.pt"),
               params=conf["num_params"], class_to_idx={"6e":0,"it":1},
               torch=conf["torch"], torchvision=conf["torchvision"], dependencies=conf["dependencies"],
               transform="full-image-square-resize/ImageNet-normalization/EXIF-transpose")
    # Commit BEFORE any audit predictions: audit cannot select temperature/TTA/epoch.
    write_json(run / "inference_config.json", inf)
    write_json(run / "calibration.json", dict(calibration_used=args.calibrate,
               raw_logloss=raw_cal, final_logloss=cal_loss, accuracy=cal_acc, temperature=temperature))
    write_json(run / "audit_metrics.json", dict(audit_evaluated=False,
               note="Not evaluated in this EMA experiment; calibration used for temperature fitting only."))
    print(f"Selected checkpoint: {read_json(run / 'best.json')}")
    print(f"Calibration raw={raw_cal:.6f} final={cal_loss:.6f} T={temperature:.6f}")
    print("Settings frozen. Calibration after-fit is a fitting score, not independent validation.")
    print("Ready for make_package.py; only the selected model.pt is deployed.")



if __name__ == "__main__":
    main()

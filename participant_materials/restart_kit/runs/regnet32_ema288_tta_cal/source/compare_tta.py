"""Compare a frozen checkpoint's original/hflip inference, then fit temperature.

TTA is chosen ONLY by select loss at T=1. Temperatures are fit on calibration.
No optimizer, model.train(), test images, or audit predictions are used by default.
"""
import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar


def fit_temperature(logits, labels, flips, metric):
    raw = metric(logits, labels, 1.0, flips)[0]
    fit = minimize_scalar(lambda t: metric(logits, labels, float(t), flips)[0],
                          bounds=(0.75, 2.0), method="bounded", options={"xatol":1e-5})
    temperature = float(fit.x) if fit.success and np.isfinite(fit.fun) and fit.fun < raw else 1.0
    return temperature, raw, metric(logits, labels, temperature, flips)[0]


def choose_tta(records):
    # Fixed tie-break: prefer the cheaper original inference.
    return "hflip" if records[1]["select_T1_logloss"] < records[0]["select_T1_logloss"] - 1e-12 else "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path(".."))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--audit", action="store_true", help="Optional: score audit AFTER choices are frozen")
    args = ap.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Choose a NEW candidate directory: {out}")
    source = run / "source"
    for name in ("model.py", "train.py", "data.py", "predict.py"):
        if not (source / name).is_file():
            raise FileNotFoundError(f"Original training source is missing: {source / name}")
    # Use the exact model/transform snapshot that produced the baseline checkpoint.
    sys.path.insert(0, str(source))
    from model import build_model
    from data import checked_rows, read_json, sha256_file, write_json
    from train import collect_logits, loader_for, metric

    conf = read_json(run / "training_config.json")
    baseline = read_json(run / "inference_config.json")
    if not (run / "completed.json").is_file():
        raise ValueError("The original training run is incomplete")
    model_hash = sha256_file(run / "model.pt")
    if model_hash != baseline["model_sha256"]:
        raise ValueError("Baseline weights do not match its frozen configuration")
    if conf["arch"] != baseline["arch"] or conf["size"] != baseline["size"]:
        raise ValueError("Training and inference architecture/resolution differ")
    rows = checked_rows(args.data, run / "splits.json")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    net = build_model(conf["arch"], pretrained=False)
    net.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=True), strict=True)
    net = net.to(device).eval()
    params = sum(p.numel() for p in net.parameters())
    if params != baseline["params"] or params > 20_000_000:
        raise ValueError("Parameter count mismatch/limit exceeded")

    cached = {}
    for role in ("select", "calibration"):
        subset = [r for r in rows if r["split"] == role]
        loader = loader_for(args.data, subset, conf["size"], args.batch)
        cached[role] = collect_logits(net, loader, device, tta=True)
        if not all(torch.isfinite(x).all() for x in cached[role] if x is not None):
            raise FloatingPointError(f"Non-finite logits in {role}")
        print(f"Finished {role}: {len(subset)} images", flush=True)

    sz, sy, sf = cached["select"]
    cz, cy, cf = cached["calibration"]
    records = []
    for tta in ("none", "hflip"):
        select_loss, acc, _, _ = metric(sz, sy, flips=sf if tta == "hflip" else None)
        temp, before, after = fit_temperature(cz, cy, cf if tta == "hflip" else None, metric)
        records.append(dict(tta=tta, select_T1_logloss=select_loss, select_T1_accuracy=acc,
                            calibration_T1_logloss=before, fitted_temperature=temp,
                            calibration_fitted_logloss=after))
    chosen = choose_tta(records)
    winner = next(r for r in records if r["tta"] == chosen)
    report = dict(
        source_run=str(run), source_model_sha256=model_hash, weights_updated=False,
        policy="TTA chosen by select loss at T=1; chosen mode's T fit on calibration in [0.75,2.0]; audit/test never choose settings",
        calibration_score_is_fit_score=True,
        comparison=records, chosen_tta=chosen, chosen_temperature=winner["fitted_temperature"],
        temperature_near_bound=winner["fitted_temperature"] < .751 or winner["fitted_temperature"] > 1.999,
        note="Calibration improvement is an in-sample fit result, not independent evidence of better generalization.")
    out.mkdir(parents=True)
    shutil.copytree(source, out / "source")
    shutil.copy2(Path(__file__), out / "source" / "compare_tta.py")
    for name in ("model.pt", "training_config.json", "splits.json", "best.json", "history.csv", "completed.json"):
        shutil.copy2(run / name, out / name)
    # Keep the original training metadata; only inference options change.
    inf = dict(baseline)
    inf.update(checkpoint="model.pt", tta=chosen, temperature=winner["fitted_temperature"],
               postprocess_selection="postprocess_comparison.json")
    write_json(out / "postprocess_comparison.json", report)
    write_json(out / "source" / "postprocess_comparison.json", report)
    with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    shutil.copy2(out / "comparison.csv", out / "source" / "comparison.csv")
    write_json(out / "calibration.json", dict(calibration_used=True, raw_logloss=winner["calibration_T1_logloss"],
               final_logloss=winner["calibration_fitted_logloss"], temperature=winner["fitted_temperature"],
               score_is_calibration_fit_score=True))
    # Freeze the candidate BEFORE optional audit evaluation.
    write_json(out / "inference_config.json", inf)
    audit_report = dict(audit_evaluated=False,
                        note="Audit deliberately not evaluated for postprocessing selection; no audit score is claimed.")
    write_json(out / "audit_metrics.json", audit_report)
    if args.audit:
        audit_rows = [r for r in rows if r["split"] == "audit"]
        loader = loader_for(args.data, audit_rows, conf["size"], args.batch)
        z, y, f = collect_logits(net, loader, device, chosen == "hflip")
        loss, acc, p, losses = metric(z, y, inf["temperature"], f)
        audit_report = dict(audit_evaluated=True, audit_n=len(y), audit_logloss=loss,
                            audit_accuracy=acc, wrong_count=int(((p >= .5) != y.numpy()).sum()),
                            note="Evaluated after settings were frozen. Do not use repeated audit evaluations to tune settings.")
        write_json(out / "audit_metrics.json", audit_report)
        print(f"Frozen candidate AUDIT: loss={loss:.6f} accuracy={acc:.2%}")
    assert sha256_file(out / "model.pt") == model_hash, "Copied weights differ"
    print("\nTTA      select(T=1)  cal(T=1)    fitted T    cal(after fit)")
    for r in records:
        print(f"{r['tta']:<8} {r['select_T1_logloss']:.6f}     {r['calibration_T1_logloss']:.6f}    "
              f"{r['fitted_temperature']:.4f}      {r['calibration_fitted_logloss']:.6f}")
    print(f"\nCHOSEN: TTA={chosen}; temperature={inf['temperature']:.6f}; params={params:,}")
    print("Calibration after-fit values are fitting scores, not independent validation scores.")
    print(f"Ready for make_package.py --run {out}")


if __name__ == "__main__":
    main()

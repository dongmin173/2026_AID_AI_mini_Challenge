"""Package frozen files, extract to a new directory, then create the official CSV on CPU."""
import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from data import read_json, sha256_file, write_json


def package(run, test_dir, output, csv_path, sample=None, include_initial=False):
    run, output, csv_path, test_dir = map(Path, (run, output, csv_path, test_dir))
    if output.exists() or csv_path.exists():
        raise FileExistsError("Choose new ZIP/CSV filenames; do not overwrite an earlier submission")
    conf = read_json(run / "inference_config.json")
    if conf["params"] > 20_000_000 or sha256_file(run / "model.pt") != conf["model_sha256"]:
        raise ValueError("Frozen model differs or exceeds parameter limit")
    with tempfile.TemporaryDirectory(prefix="restart_package_") as tmp:
        folder = Path(tmp) / "package"
        folder.mkdir()
        for src in sorted((run / "source").iterdir()):
            if src.is_file():
                shutil.copy2(src, folder / src.name)
        names = ["model.pt", "inference_config.json", "training_config.json", "splits.json",
                 "best.json", "history.csv", "completed.json", "calibration.json", "audit_metrics.json"]
        if include_initial:
            names.append("initial_model.pt")
        for name in names:
            shutil.copy2(run / name, folder / name)
        (folder / "requirements_cpu.txt").write_text(
            f"torch=={conf['torch'].split('+')[0]}\n"
            f"torchvision=={conf['torchvision'].split('+')[0]}\n"
            + "".join(f"{name}=={v}\n" for name, v in conf["dependencies"].items()), encoding="utf-8")
        (folder / "RUN_INFERENCE.txt").write_text(
            "Offline CPU reproduction (no weight downloads):\n"
            "python predict.py --test_dir PATH_TO_TEST --output submission.csv\n\n"
            "Train code: train.py, model.py, data.py; exact options: training_config.json.\n"
            "Split identities and input checksums: splits.json.\n"
            "Only model.pt is used for inference. If initial_model.pt is included,\n"
            "it is training initialization only, NOT an ensemble member.\n"
            "Class 6e=0, it=1. Matching framework versions are recorded.\n"
            "Measure the full command on Google Colab CPU before final submission.\n", encoding="utf-8")
        archive = Path(tmp) / "verified_package.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as z:
            for p in sorted(folder.iterdir()):
                if p.is_file():
                    z.write(p, p.name)
        if archive.stat().st_size > 200_000_000:
            raise ValueError("ZIP exceeds 200 MB")
        extracted = Path(tmp) / "extracted"
        with zipfile.ZipFile(archive) as z:
            z.extractall(extracted)
        reproduced = Path(tmp) / "submission.csv"
        command = [sys.executable, str(extracted / "predict.py"),
                   "--test-dir", str(test_dir.resolve()), "--output", str(reproduced)]
        if sample:
            command.extend(["--sample", str(Path(sample).resolve())])
        start = time.perf_counter()
        subprocess.run(command, cwd=extracted, check=True)
        seconds = time.perf_counter() - start
        if seconds > 600:
            raise RuntimeError(f"End-to-end process time {seconds:.1f}s exceeds 600s")
        with reproduced.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ["id", "prob"]:
                raise ValueError("Invalid CSV header")
            records = list(reader)
        expected_ids = {p.stem for p in test_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        if len(records) != 500 or len({r["id"] for r in records}) != 500 or {r["id"] for r in records} != expected_ids:
            raise ValueError("Wrong test ID coverage")
        if not all(0 <= float(r["prob"]) <= 1 for r in records):
            raise ValueError("Invalid probability")
        output.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, output)
        shutil.copy2(reproduced, csv_path)
        write_json(output.with_suffix(".verification.json"), dict(
            local_cpu_seconds=seconds, params=conf["params"],
            zip_bytes=output.stat().st_size, csv_sha256=sha256_file(csv_path),
            zip_sha256=sha256_file(output),
            colab_600_seconds_verified=False))
    print(f"Created matched pair: {csv_path} + {output}")
    print("CSV was generated from this extracted ZIP. Google Colab CPU timing is still required.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--sample", type=Path)
    ap.add_argument("--include-initial", action="store_true")
    args = ap.parse_args()
    package(args.run, args.test_dir, args.output, args.csv, args.sample, args.include_initial)


if __name__ == "__main__":
    main()

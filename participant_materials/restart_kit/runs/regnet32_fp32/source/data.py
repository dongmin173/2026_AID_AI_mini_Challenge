"""Train-only inventory, fixed split identities, and shared datasets."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def image_files(path):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Image folder not found: {path.resolve()}")
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare_split(data_root, output, seed=20261107, expected=2906):
    """Only inspect train/. Exact decoded duplicates cannot cross partitions.

    Near-duplicate dHash matches are review candidates, NOT automatic identity.
    User-provided shooting-session groups could be substituted in future work.
    """
    root, output = Path(data_root), Path(output)
    if output.exists():
        raise FileExistsError(f"Split already exists; reuse it: {output}")
    rows, dhashes = [], []
    for label, folder in enumerate(("6e", "it")):
        for path in image_files(root / "train" / folder):
            with Image.open(path) as im:
                rgb = ImageOps.exif_transpose(im).convert("RGB")
                h = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                thumb = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
                bits = (thumb[:, 1:] > thumb[:, :-1]).reshape(-1)
                dhashes.append(int.from_bytes(np.packbits(bits).tobytes(), "big"))
                width, height = rgb.size
            rows.append(dict(path=path.relative_to(root).as_posix(), label=label,
                             group=h, sha256=sha256_file(path), width=width, height=height))
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} TRAIN images, found {len(rows)}; check --data")
    groups_to_labels = defaultdict(set)
    for r in rows:
        groups_to_labels[r["group"]].add(r["label"])
    conflicts = [g for g, ys in groups_to_labels.items() if len(ys) > 1]
    if conflicts:
        raise ValueError("Identical images have conflicting class labels. Inspect train data first.")
    ys = np.array([r["label"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    # Four roles, approximately 70/10/10/10. All random choices fixed before training.
    splitter = StratifiedGroupKFold(10, shuffle=True, random_state=seed)
    for fold, (_, idx) in enumerate(splitter.split(np.zeros(len(rows)), ys, groups)):
        role = {0: "select", 1: "calibration", 2: "audit"}.get(fold, "train")
        for i in idx:
            rows[i]["split"] = role
    counts = {role: dict(Counter(str(r["label"]) for r in rows if r["split"] == role))
              for role in ("train", "select", "calibration", "audit")}
    for role, c in counts.items():
        if not all(int(c.get(str(y), 0)) > 0 for y in (0, 1)):
            raise ValueError(f"Split lacks a class: {role}; inspect duplicate groups")
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs_path = output.with_name(output.stem + "_near_duplicates.csv")
    pairs = 0
    with pairs_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path_a", "path_b", "hamming", "same_pixels", "split_a", "split_b"])
        for i, h in enumerate(dhashes):
            for j in range(i):
                distance = (h ^ dhashes[j]).bit_count()
                if distance <= 3:
                    writer.writerow([rows[j]["path"], rows[i]["path"], distance,
                                     rows[j]["group"] == rows[i]["group"],
                                     rows[j]["split"], rows[i]["split"]])
                    pairs += 1
    result = dict(schema_version=1, seed=seed, class_to_idx={"6e": 0, "it": 1},
                  counts=counts, rows=rows, exact_duplicate_groups=sum(
                      n > 1 for n in Counter(groups).values()), near_duplicate_candidates=pairs)
    write_json(output, result)
    print(json.dumps(counts, indent=2))
    print(f"Exact duplicate groups: {result['exact_duplicate_groups']}; near candidates: {pairs}")
    print(f"Saved {output}; candidates for visual review: {pairs_path}")
    print("Near candidates are NOT proof of duplicates. Similar views may still cross splits.")


def checked_rows(root, split_file):
    root = Path(root)
    obj = read_json(split_file)
    rows = obj["rows"]
    if obj["class_to_idx"] != {"6e": 0, "it": 1}:
        raise ValueError("Wrong label mapping")
    inventory = {p.relative_to(root).as_posix() for folder in ("6e", "it")
                 for p in image_files(root / "train" / folder)}
    if inventory != {r["path"] for r in rows} or len(rows) != len(inventory):
        raise ValueError("Train inventory differs from saved split; do not reuse altered data")
    seen = defaultdict(set)
    for r in rows:
        path = Path(r["path"])
        if path.is_absolute() or ".." in path.parts or path.parts[:2] != ("train", ("6e", "it")[r["label"]]):
            raise ValueError(f"Invalid train path: {path}")
        if sha256_file(root / path) != r["sha256"]:
            raise ValueError(f"Image content changed since split creation: {path}")
        seen[r["group"]].add(r["split"])
    if any(len(s) != 1 for s in seen.values()):
        raise ValueError("Exact duplicates cross partitions")
    return rows


class TrainImages:
    # A map-style Dataset; no inheritance needed by PyTorch DataLoader.
    def __init__(self, root, rows, transform):
        self.root, self.rows, self.transform = Path(root), rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from model import load_rgb
        r = self.rows[i]
        return self.transform(load_rgb(self.root / r["path"])), r["label"]

"""Reconstruct the YOLO detection dataset from the PUBLISHED files — no Label Studio needed.

Reads the source images (data/corseacare/), the YOLO labels (annotations/<stem>.txt) and the
manifest (samples.csv: which images + the train/val split), and writes
data/ls_export_yolo/{images,labels}/{train,val} + data.yaml — exactly what tile_dataset.py
consumes. This lets anyone reproduce training from the public repo without the original Label
Studio project.

    .venv/bin/python scripts/build_yolo_dataset.py
"""
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "corseacare"
ANN = ROOT / "annotations"
OUT = ROOT / "data" / "ls_export_yolo"
CLASSES = ["fragment", "fibre", "film", "mousse", "pellet", "autre"]


def find_image(stem):
    for ext in (".jpg", ".JPG", ".jpeg", ".png"):
        p = IMAGES / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    man = {}
    if (ROOT / "samples.csv").exists():
        man = {r["image"]: r for r in csv.DictReader(open(ROOT / "samples.csv"))}
    if OUT.exists():
        shutil.rmtree(OUT)
    n = {"train": 0, "val": 0}
    for lbl in sorted(ANN.glob("*.txt")):
        img = find_image(lbl.stem)
        if not img:
            print(f"skip {lbl.stem}: image not found in {IMAGES}")
            continue
        split = (man.get(img.name, {}).get("split") or "train").strip() or "train"
        if split not in ("train", "val"):
            split = "train"
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy(img, OUT / "images" / split / img.name)
        shutil.copy(lbl, OUT / "labels" / split / f"{lbl.stem}.txt")
        n[split] += 1
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnames: [{', '.join(CLASSES)}]\n")
    print(f"{n['train']} train / {n['val']} val images -> {OUT}")
    print("next:  python scripts/tile_dataset.py  &&  python scripts/train_tiles.py")


if __name__ == "__main__":
    main()

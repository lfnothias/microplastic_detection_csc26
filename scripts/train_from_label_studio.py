"""Fine-tune YOLO11n on the M4 from a Label Studio **YOLO export** of the real photos.

Usage:
    .venv/bin/python scripts/train_from_label_studio.py <label_studio_YOLO_export_dir>

The export dir is expected to contain images/, labels/, and classes.txt (Label Studio's
'YOLO' export format). This makes an 80/20 train/val split, writes a data.yaml using the
exported class names, and fine-tunes on MPS. Weights land in data/runs/corseacare_real/.
"""
import sys
import shutil
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(export_dir: str):
    export = Path(export_dir)
    classes = ([c for c in (export / "classes.txt").read_text().splitlines() if c.strip()]
               if (export / "classes.txt").exists() else ["fragment"])
    imgs = sorted(p for p in (export / "images").glob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not imgs:
        print(f"No images in {export/'images'} — is this a Label Studio YOLO export?")
        return
    out = export / "_yolo_split"
    rng = random.Random(0)
    val_idx = set(rng.sample(range(len(imgs)), max(1, len(imgs) // 5)))
    for i, img in enumerate(imgs):
        split = "val" if i in val_idx else "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy(img, out / "images" / split / img.name)
        lbl = export / "labels" / f"{img.stem}.txt"
        (out / "labels" / split / f"{img.stem}.txt").write_text(lbl.read_text() if lbl.exists() else "")
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnames: [{', '.join(classes)}]\n")
    print(f"split: {len(imgs)} images ({len(val_idx)} val), classes={classes}")

    from ultralytics import YOLO
    m = YOLO("yolo11n.pt")
    m.train(data=str(out / "data.yaml"), epochs=80, imgsz=640, batch=8, device="mps",
            project=str(REPO / "data" / "runs"), name="corseacare_real", exist_ok=True, verbose=False)
    metrics = m.val(data=str(out / "data.yaml"), device="mps", verbose=False)
    print("VAL mAP50:", float(metrics.box.map50), "| mAP50-95:", float(metrics.box.map))
    print("BEST:", REPO / "data" / "runs" / "corseacare_real" / "weights" / "best.pt")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: train_from_label_studio.py <label_studio_YOLO_export_dir>")
        sys.exit(1)
    main(sys.argv[1])

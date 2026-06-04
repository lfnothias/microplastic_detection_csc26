"""Convert the synthetic test dataset's ground_truth.json (bboxes) into a YOLO-det
dataset on disk, for local fine-tuning. Single class: fragment (id 0).

Output goes under data/test_run_yolo/ (gitignored via the /data/ rule).
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "test_run"
OUT = ROOT / "data" / "test_run_yolo"
IMG_W = IMG_H = 480


def main():
    gt = json.loads((SRC / "ground_truth.json").read_text())
    names = sorted(gt)
    n_val = max(1, len(names) // 5)  # ~20% validation
    for i, name in enumerate(names):
        split = "val" if i < n_val else "train"
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy(SRC / "images" / name, OUT / "images" / split / name)
        lines = []
        for p in gt[name]["particles"]:
            x1, y1, x2, y2 = p["bbox"]
            cx = ((x1 + x2) / 2) / IMG_W
            cy = ((y1 + y2) / 2) / IMG_H
            w = (x2 - x1) / IMG_W
            h = (y2 - y1) / IMG_H
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (OUT / "labels" / split / f"{Path(name).stem}.txt").write_text("\n".join(lines))
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnames: [fragment]\n"
    )
    print(f"wrote YOLO dataset to {OUT} ({len(names)} images, {n_val} val)")


if __name__ == "__main__":
    main()

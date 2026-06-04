"""Crop candidate particles into numbered montage grids for Claude-vision classification.

Reads a pre-annotation `ls_tasks.json` (saturation or SAM2), crops each candidate box
(with padding) from the full-res photo, and lays the crops out in numbered grids so a
vision model can classify each cell. Writes:
  <preann>/crops/montage_<image>_<page>.png   numbered grids (cells labelled 0,1,2,…)
  <preann>/crops/index.json                   {montage_file: [{"id", "image", "xyxy"}, …]}

Usage:
    .venv/bin/python scripts/make_candidate_crops.py [preann_dir]
        preann_dir defaults to data/corseacare_preann (the saturation set).
"""
import json
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "corseacare"
CELL = 120          # montage cell size (px)
COLS = 6            # grid columns
PER_PAGE = 36       # crops per montage page (COLS x 6)
PAD = 0.25          # crop padding as fraction of box size


def crop(img, xyxy):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = xyxy
    pw, ph = int((x2 - x1) * PAD), int((y2 - y1) * PAD)
    x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
    x2, y2 = min(W, x2 + pw), min(H, y2 + ph)
    c = img[y1:y2, x1:x2]
    return c if c.size else np.zeros((CELL, CELL, 3), np.uint8)


def main(preann_dir):
    preann = Path(preann_dir)
    tasks = json.loads((preann / "ls_tasks.json").read_text())
    out = preann / "crops"; out.mkdir(parents=True, exist_ok=True)
    index = {}
    for task in tasks:
        name = task["data"]["image"].rsplit("/", 1)[-1]
        img = cv2.imread(str(IMAGES / name))
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = []
        for r in task["predictions"][0]["result"]:
            v = r["value"]
            x1, y1 = v["x"] / 100 * W, v["y"] / 100 * H
            x2, y2 = x1 + v["width"] / 100 * W, y1 + v["height"] / 100 * H
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
        for page in range((len(boxes) + PER_PAGE - 1) // PER_PAGE):
            chunk = boxes[page * PER_PAGE:(page + 1) * PER_PAGE]
            rows = (len(chunk) + COLS - 1) // COLS
            canvas = np.full((rows * CELL, COLS * CELL, 3), 30, np.uint8)
            entries = []
            for i, b in enumerate(chunk):
                cell = cv2.resize(crop(img, b), (CELL, CELL))
                rr, cc = divmod(i, COLS)
                canvas[rr * CELL:(rr + 1) * CELL, cc * CELL:(cc + 1) * CELL] = cell
                cv2.putText(canvas, str(i), (cc * CELL + 3, rr * CELL + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                entries.append({"id": i, "image": name, "xyxy": b})
            mfile = f"montage_{Path(name).stem}_{page}.png"
            cv2.imwrite(str(out / mfile), canvas)
            index[mfile] = entries
    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"{len(index)} montage(s), "
          f"{sum(len(v) for v in index.values())} crops -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "corseacare_preann"))

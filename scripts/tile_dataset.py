"""Slice the LS-export YOLO dataset (data/ls_export_yolo) into overlapping tiles, remapping
boxes per tile (SAHI-style). Preserves the photo-level train/val split (no tile leakage).
Keeps every tile that contains a box, plus a fraction of empty tiles as negatives.

    .venv/bin/python scripts/tile_dataset.py

Output: data/ls_tiles_yolo/{images,labels}/{train,val} + data.yaml.
"""
import random
from pathlib import Path
import numpy as np
import cv2

from corseacare.tiling import tile_origins, remap_boxes_to_tile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "ls_export_yolo"
OUT = ROOT / "data" / "ls_tiles_yolo"
CLASSES = ["fragment", "fibre", "film", "mousse", "pellet", "autre"]
TILE, OVERLAP, MIN_VISIBLE, EMPTY_KEEP = 640, 0.3, 0.3, 0.15


def load_boxes(txt):
    if not txt.exists():
        return []
    out = []
    for ln in txt.read_text().splitlines():
        p = ln.split()
        if len(p) == 5:
            out.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return out


def main():
    rng = random.Random(0)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        n_tiles = n_box_tiles = n_boxes = 0
        for img_path in sorted((SRC / "images" / split).glob("*")):
            img = cv2.imread(str(img_path))
            H, W = img.shape[:2]
            boxes = load_boxes(SRC / "labels" / split / f"{img_path.stem}.txt")
            for (x0, y0) in tile_origins(W, H, TILE, OVERLAP):
                tb = remap_boxes_to_tile(boxes, W, H, x0, y0, TILE, MIN_VISIBLE)
                if not tb and rng.random() > EMPTY_KEEP:
                    continue
                tile = img[y0:y0 + TILE, x0:x0 + TILE]
                th, tw = tile.shape[:2]
                if th < TILE or tw < TILE:           # edge guard (origins are flush, but be safe)
                    pad = np.zeros((TILE, TILE, 3), np.uint8); pad[:th, :tw] = tile; tile = pad
                stem = f"{img_path.stem}_{x0}_{y0}"
                cv2.imwrite(str(OUT / "images" / split / f"{stem}.jpg"), tile)
                (OUT / "labels" / split / f"{stem}.txt").write_text(
                    "\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in tb))
                n_tiles += 1
                n_box_tiles += 1 if tb else 0
                n_boxes += len(tb)
        print(f"{split}: {n_tiles} tiles ({n_box_tiles} with boxes, {n_boxes} boxes)")
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnames: [{', '.join(CLASSES)}]\n")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

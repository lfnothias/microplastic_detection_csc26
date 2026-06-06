"""Tiled (SAHI) inference over the full photos: slice into tiles, detect per tile, offset
boxes back, merge with NMS, draw overlay + per-class counts. Use the tiles-trained weights.

    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/predict_tiled.py [weights] [conf]
"""
import sys
import glob
from collections import Counter
from pathlib import Path
import numpy as np
import cv2

from corseacare.config import Config
from corseacare.tiling import tile_origins, offset_boxes_from_tile, nms
from corseacare.sieve import detect_sieve_circle, keep_inside_circle

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "corseacare"
OUT = ROOT / "data" / "corseacare_pred_tiled"
TILE, OVERLAP = 640, 0.5
ROI_MARGIN = 1.0      # keep detections up to the sieve rim (drops rim/tray FPs beyond it)
MAX_BOX_FRAC = 0.04   # drop boxes larger than 4% of the image (oversized FPs / plaques)
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 165, 255), (255, 0, 255), (160, 160, 160)]


def main(weights, conf):
    from ultralytics import YOLO
    cfg = Config()
    model = YOLO(weights)
    (OUT / "overlays").mkdir(parents=True, exist_ok=True)
    rows, per_class = [], Counter()
    exts = ("*.JPG", "*.jpg", "*.jpeg", "*.JPEG", "*.png", "*.PNG")
    paths = sorted({p for e in exts for p in glob.glob(str(IMAGES / e))})
    for p in paths:
        img = cv2.imread(p)
        H, W = img.shape[:2]
        dets = []
        for (x0, y0) in tile_origins(W, H, TILE, OVERLAP):
            tile = img[y0:y0 + TILE, x0:x0 + TILE]
            th, tw = tile.shape[:2]
            if th < TILE or tw < TILE:
                pad = np.zeros((TILE, TILE, 3), np.uint8); pad[:th, :tw] = tile; tile = pad
            res = model.predict(tile, conf=conf, verbose=False)[0]
            td = [(int(b.cls.item()), float(b.conf.item()), *[float(v) for v in b.xyxy[0].tolist()])
                  for b in res.boxes]
            dets += offset_boxes_from_tile(td, x0, y0)
        merged = nms(dets, iou_thr=0.5)
        circle = detect_sieve_circle(img)              # gate out rim / table / tool false positives
        n_raw = len(merged)
        merged = keep_inside_circle(merged, circle, margin=ROI_MARGIN)
        amax = H * W * MAX_BOX_FRAC
        merged = [d for d in merged if (d[4] - d[2]) * (d[5] - d[3]) <= amax]
        ov = img.copy()
        cx, cy, r = (int(v) for v in circle)
        cv2.circle(ov, (cx, cy), r, (0, 255, 255), 2)  # show the ROI we kept
        for (cls, cf, x1, y1, x2, y2) in merged:
            name = cfg.classes[cls] if cls < len(cfg.classes) else str(cls)
            cv2.rectangle(ov, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[cls % len(COLORS)], 3)
            per_class[name] += 1
            rows.append({"image": Path(p).name, "class": name, "conf": round(cf, 3),
                         "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)})
        s = 1400 / max(H, W)
        cv2.imwrite(str(OUT / "overlays" / Path(p).name), cv2.resize(ov, (int(W * s), int(H * s))))
        print(f"{Path(p).name}: {len(merged)} particles ({n_raw - len(merged)} dropped outside sieve)")
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / "counts.csv", index=False)
    print("per-class totals:", dict(per_class))
    print(f"overlays + counts.csv -> {OUT}")


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "runs" / "tiles_seed" / "weights" / "best.pt")
    c = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    main(w, c)

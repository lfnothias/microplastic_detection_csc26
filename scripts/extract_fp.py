"""Extract a model's false positives on the held-out sieve = candidate MISSED annotations.

Runs TiledDetector on each validation image (split==val in samples.csv), matches predictions to
the ground-truth boxes by IoU, and the UNMATCHED predictions become candidate boxes for
re-annotation (the annotation is incomplete, so many of these are real un-boxed particles).
Crops them into numbered montages for vision-model validation and saves their boxes.

    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/extract_fp.py [weights] [conf] [iou]

Output: data/reannot_B/{montage_fp_*.png, index.json, fp_boxes.json}
"""
import sys
import csv
import json
from pathlib import Path
import numpy as np
import cv2

from corseacare.config import Config
from corseacare.pipeline.detector import TiledDetector
from corseacare.tiling import _iou

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "corseacare"
GT_DIR = ROOT / "data" / "ls_export_yolo" / "labels" / "val"
OUT = ROOT / "data" / "reannot_B"
CLASSES = ["fragment", "fibre", "film", "mousse", "pellet", "autre"]
CELL, COLS, PER_PAGE, PAD = 160, 6, 36, 0.3


def load_gt(stem, W, H):
    f = GT_DIR / f"{stem}.txt"
    out = []
    if not f.exists():
        return out
    for ln in f.read_text().splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        c, cx, cy, w, h = int(p[0]), *(float(x) for x in p[1:])
        out.append((CLASSES[c], ((cx - w / 2) * W, (cy - h / 2) * H,
                                 (cx + w / 2) * W, (cy + h / 2) * H)))
    return out


def crop(img, xyxy):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = xyxy
    pw, ph = int((x2 - x1) * PAD), int((y2 - y1) * PAD)
    x1, y1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
    x2, y2 = min(W, int(x2 + pw)), min(H, int(y2 + ph))
    c = img[y1:y2, x1:x2]
    return c if c.size else np.zeros((CELL, CELL, 3), np.uint8)


def main(weights, conf, iou_thr):
    cfg = Config()
    det = TiledDetector(weights, cfg.classes, conf=conf, tile=cfg.tile_size,
                        overlap=cfg.tile_overlap, roi_gate=cfg.roi_gate)
    val = [r["image"] for r in csv.DictReader(open(ROOT / "samples.csv"))
           if (r.get("split") or "").strip() == "val"]
    val = [im for im in val if (GT_DIR / f"{Path(im).stem}.txt").exists()]
    OUT.mkdir(parents=True, exist_ok=True)

    fps = []  # (image, xyxy, pred_class, conf)
    for im in val:
        img = cv2.imread(str(IMAGES / im))
        H, W = img.shape[:2]
        dets = det.predict(img)
        gts = load_gt(Path(im).stem, W, H)
        used = set()
        for d in sorted(dets, key=lambda d: -d.confidence):
            best, bj = iou_thr, -1
            for j, (gn, gb) in enumerate(gts):
                if j in used:
                    continue
                v = _iou(d.xyxy, gb)
                if v >= best:
                    best, bj = v, j
            if bj < 0:
                fps.append((im, [int(x) for x in d.xyxy], d.class_name, round(float(d.confidence), 3)))
            else:
                used.add(bj)
        print(f"{im}: {len(dets)} préd, {len(gts)} GT, {sum(1 for f in fps if f[0] == im)} FP")

    index = {}
    for pg in range((len(fps) + PER_PAGE - 1) // PER_PAGE):
        chunk = fps[pg * PER_PAGE:(pg + 1) * PER_PAGE]
        rows = (len(chunk) + COLS - 1) // COLS
        canvas = np.full((rows * CELL, COLS * CELL, 3), 30, np.uint8)
        ents = []
        for i, (im, box, pc, cf) in enumerate(chunk):
            img = cv2.imread(str(IMAGES / im))
            cell = cv2.resize(crop(img, box), (CELL, CELL))
            rr, cc = divmod(i, COLS)
            canvas[rr * CELL:(rr + 1) * CELL, cc * CELL:(cc + 1) * CELL] = cell
            cv2.putText(canvas, str(i), (cc * CELL + 3, rr * CELL + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            ents.append({"id": i, "image": im, "xyxy": box, "pred_class": pc, "conf": cf})
        mf = f"montage_fp_{pg}.png"
        cv2.imwrite(str(OUT / mf), canvas)
        index[mf] = ents
    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    (OUT / "fp_boxes.json").write_text(json.dumps(fps, indent=2))
    print(f"\n{len(fps)} faux positifs -> {len(index)} montage(s) dans {OUT}")


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data/runs/tiles_v6/weights/best.pt")
    c = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    i = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    main(w, c, i)

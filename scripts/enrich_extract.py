"""Enrichment step 1 — extract candidate MISSED particles on every annotated image.

For each image that has a GT label in data/ls_export_yolo (train or val), run the tiled detector,
keep predictions that do NOT match any GT box (IoU) — these are candidate un-annotated particles —
and crop them into numbered montages for a strict vision-model pass. GT is also copied so the
later merge can write enriched labels.

    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/enrich_extract.py [weights] [conf] [iou]
Output: data/enrich/{montages/*.png, index.json, gt/<stem>.txt, dims.json}
"""
import sys
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import cv2

from corseacare.config import Config
from corseacare.pipeline.detector import TiledDetector
from corseacare.tiling import _iou

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "corseacare"
LBL = ROOT / "data" / "ls_export_yolo" / "labels"
OUT = ROOT / "data" / "enrich"
CLASSES = ["fragment", "fibre", "film", "mousse", "pellet", "autre"]
CELL, COLS, PER_PAGE, PAD = 160, 6, 36, 0.3


def find_label(stem):
    for sp in ("train", "val"):
        f = LBL / sp / f"{stem}.txt"
        if f.exists():
            return f
    return None


def load_gt(stem, W, H):
    f = find_label(stem)
    out = []
    if not f:
        return out
    for ln in f.read_text().splitlines():
        p = ln.split()
        if len(p) == 5:
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
    rows = list(csv.DictReader(open(ROOT / "samples.csv")))
    stem2img = {Path(r["image"]).stem: r["image"] for r in rows}
    stem2sid = {Path(r["image"]).stem: r["sample_id"] for r in rows}
    only = (sys.argv[4] if len(sys.argv) > 4 else "TAMIS")   # sample_id prefix filter ("" = all)
    stems = sorted({f.stem for sp in ("train", "val") for f in (LBL / sp).glob("*.txt")
                    if stem2sid.get(f.stem, "").startswith(only)})
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "montages").mkdir(parents=True)
    (OUT / "gt").mkdir(parents=True)

    cands = []   # (image, xyxy, pred_class, conf)
    dims = {}
    for stem in stems:
        im = stem2img.get(stem)
        if not im or not (IMAGES / im).exists():
            continue
        img = cv2.imread(str(IMAGES / im))
        H, W = img.shape[:2]
        dims[im] = [W, H]
        shutil.copy(find_label(stem), OUT / "gt" / f"{stem}.txt")
        gts = load_gt(stem, W, H)
        dets = det.predict(img)
        used = set()
        n = 0
        for d in sorted(dets, key=lambda d: -d.confidence):
            best, bj = iou_thr, -1
            for j, (gn, gb) in enumerate(gts):
                if j in used:
                    continue
                v = _iou(d.xyxy, gb)
                if v >= best:
                    best, bj = v, j
            if bj < 0:
                cands.append((im, [int(x) for x in d.xyxy], d.class_name, round(float(d.confidence), 3)))
                n += 1
            else:
                used.add(bj)
        print(f"{im}: {len(dets)} préd, {len(gts)} GT, {n} candidats")

    index = {}
    for pg in range((len(cands) + PER_PAGE - 1) // PER_PAGE):
        chunk = cands[pg * PER_PAGE:(pg + 1) * PER_PAGE]
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
        mf = f"montage_{pg}.png"
        cv2.imwrite(str(OUT / "montages" / mf), canvas)
        index[mf] = ents
    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    (OUT / "dims.json").write_text(json.dumps(dims, indent=2))
    print(f"\n{len(cands)} candidats -> {len(index)} montage(s) dans {OUT}/montages")


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data/runs/tiles_v6/weights/best.pt")
    c = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    i = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    main(w, c, i)

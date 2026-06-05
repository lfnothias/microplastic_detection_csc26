"""Plastique vs matière organique — métriques binaires sur le(s) tamis held-out.

Collapse les 5 morphotypes plastique (fragment/fibre/film/mousse/pellet) -> 'plastique' et
'autre' -> 'organique'. Tourne la détection tuilée (TiledDetector + ROI-gating) sur chaque
image de validation (split==val dans samples.csv), apparie aux annotations (GT) par IoU, puis
reporte la matrice de confusion binaire, précision/rappel/F1, et le taux de confusion
plastique<->organique parmi les particules détectées.

    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/eval_binary.py [weights] [conf] [iou]
"""
import sys
import csv
from pathlib import Path
import cv2

from corseacare.config import Config
from corseacare.pipeline.detector import TiledDetector
from corseacare.tiling import _iou

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "corseacare"
GT_DIR = ROOT / "data" / "ls_export_yolo" / "labels" / "val"
CLASSES = ["fragment", "fibre", "film", "mousse", "pellet", "autre"]
PLASTIC = {"fragment", "fibre", "film", "mousse", "pellet"}


def is_plastic(cls_name):
    return cls_name in PLASTIC


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
        x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
        x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
        out.append((CLASSES[c], (x1, y1, x2, y2)))
    return out


def main(weights, conf, iou_thr):
    cfg = Config()
    det = TiledDetector(weights, cfg.classes, conf=conf, tile=cfg.tile_size,
                        overlap=cfg.tile_overlap, roi_gate=cfg.roi_gate)
    val_imgs = [r["image"] for r in csv.DictReader(open(ROOT / "samples.csv"))
                if (r.get("split") or "").strip() == "val"]
    val_imgs = [im for im in val_imgs if (GT_DIR / f"{Path(im).stem}.txt").exists()]
    if not val_imgs:
        print("aucune image de val annotée trouvée (split==val + label présent)"); return

    # binary tallies: GT plastic/organic x pred plastic/organic, + missed (FN) + FP
    Pp = Po = Op = Oo = 0          # GT(P/O) x pred(P/O) parmi appariés
    miss_P = miss_O = fp_P = fp_O = 0
    for im in val_imgs:
        img = cv2.imread(str(IMAGES / im))
        H, W = img.shape[:2]
        dets = det.predict(img)
        preds = [(d.class_name, d.xyxy, d.confidence) for d in dets]
        gts = load_gt(Path(im).stem, W, H)
        used = set()
        for pname, pbox, _ in sorted(preds, key=lambda d: -d[2]):
            best, bj = iou_thr, -1
            for j, (gname, gbox) in enumerate(gts):
                if j in used:
                    continue
                v = _iou(pbox, gbox)
                if v >= best:
                    best, bj = v, j
            if bj < 0:
                fp_P += is_plastic(pname); fp_O += not is_plastic(pname); continue
            used.add(bj)
            gp, pp = is_plastic(gts[bj][0]), is_plastic(pname)
            Pp += gp and pp; Po += gp and not pp; Op += (not gp) and pp; Oo += (not gp) and not pp
        for j, (gname, _) in enumerate(gts):
            if j not in used:
                miss_P += is_plastic(gname); miss_O += not is_plastic(gname)
        print(f"  {im}: {len(gts)} GT, {len(preds)} préd.")

    gtP, gtO = Pp + Po + miss_P, Oo + Op + miss_O
    print(f"\n=== Binaire PLASTIQUE vs ORGANIQUE — held-out: {val_imgs} (conf={conf}, IoU={iou_thr}) ===")
    print(f"GT: plastique={gtP}  organique={gtO}")
    print("\nMatrice (parmi particules détectées & appariées) :")
    print(f"                 préd:PLASTIQUE   préd:ORGANIQUE")
    print(f"  GT PLASTIQUE        {Pp:6d}          {Po:6d}")
    print(f"  GT ORGANIQUE        {Op:6d}          {Oo:6d}")
    print(f"  (ratés/non détectés: plastique={miss_P}, organique={miss_O}; "
          f"faux positifs: plastique={fp_P}, organique={fp_O})")

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f
    # plastique: TP=Pp, FP=préd plastique faux (Op + fp_P), FN=plastique manqué (Po + miss_P)
    pP = prf(Pp, Op + fp_P, Po + miss_P)
    pO = prf(Oo, Po + fp_O, Op + miss_O)
    print("\nMétriques binaires (détection + classification) :")
    print(f"  PLASTIQUE  : précision={pP[0]:.2f}  rappel={pP[1]:.2f}  F1={pP[2]:.2f}")
    print(f"  ORGANIQUE  : précision={pO[0]:.2f}  rappel={pO[1]:.2f}  F1={pO[2]:.2f}")
    matched = Pp + Po + Op + Oo
    if matched:
        print(f"\nParmi les particules détectées, classification binaire correcte : "
              f"{(Pp + Oo) / matched * 100:.1f}%  ({Pp + Oo}/{matched})")


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data/runs/tiles_v6/weights/best.pt")
    c = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    i = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    main(w, c, i)

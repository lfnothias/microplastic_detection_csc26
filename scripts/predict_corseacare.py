"""Run the trained 6-class model over the real CorSeaCare photos: overlays + per-particle CSV
+ per-class counts. Uses the corseacare pipeline (UltralyticsDetector + box-mask segmenter +
measure). Default weights = the real-seed model.

    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/predict_corseacare.py [weights] [conf]
"""
import sys
import glob
from collections import Counter
from pathlib import Path
import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "corseacare"
OUT = ROOT / "data" / "corseacare_pred"


def main(weights, conf):
    from corseacare.config import Config
    from corseacare.pipeline.detector import UltralyticsDetector
    from corseacare.pipeline.segment_sam2 import FakeSegmenter
    from corseacare.pipeline.sequential import Pipeline
    from corseacare.viz import draw_overlay

    cfg = Config()
    pipe = Pipeline(UltralyticsDetector(weights, cfg.classes, conf=conf), FakeSegmenter(), cfg.mm_per_px)
    (OUT / "overlays").mkdir(parents=True, exist_ok=True)
    rows, per_class = [], Counter()
    for p in sorted(glob.glob(str(IMAGES / "*.JPG"))):
        img = cv2.imread(p)
        r = pipe.run(img)
        cv2.imwrite(str(OUT / "overlays" / Path(p).name), draw_overlay(img, r["detections"], r["masks"]))
        for rec in r["records"]:
            rec["image"] = Path(p).name
            rows.append(rec)
            per_class[rec["class_name"]] += 1
        print(f"{Path(p).name}: {r['count']} particles")
    pd.DataFrame(rows).to_csv(OUT / "counts.csv", index=False)
    print("per-class totals:", dict(per_class))
    print(f"overlays + counts.csv -> {OUT}")


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "runs" / "real_seed" / "weights" / "best.pt")
    c = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    main(w, c)

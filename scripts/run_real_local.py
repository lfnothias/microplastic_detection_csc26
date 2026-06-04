"""Run the REAL CorSeaCare pipeline (trained YOLO11 + FakeSegmenter box masks) on the
synthetic test dataset and compare counts to ground truth. Uses the locally fine-tuned
weights from train_local.py. Shows whether the trained RGB model catches the blue
particles the classical brightness detector missed.
"""
import json
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "test_run"
WEIGHTS = ROOT / "data" / "runs" / "local_poc" / "weights" / "best.pt"


def main():
    from corseacare.pipeline.detector import UltralyticsDetector
    from corseacare.pipeline.segment_sam2 import FakeSegmenter
    from corseacare.pipeline.sequential import Pipeline
    from corseacare.metrics import counting_error
    from corseacare.viz import draw_overlay

    gt = json.loads((SRC / "ground_truth.json").read_text())
    det = UltralyticsDetector(str(WEIGHTS), ["fragment"], conf=0.25)
    pipe = Pipeline(det, FakeSegmenter(), mm_per_px=0.1)
    out = SRC / "overlays_real"; out.mkdir(exist_ok=True)
    pred, true = [], []
    for name in sorted(gt):
        img = cv2.imread(str(SRC / "images" / name))
        r = pipe.run(img)
        cv2.imwrite(str(out / name), draw_overlay(img, r["detections"], r["masks"]))
        pred.append(r["count"]); true.append(gt[name]["count"])
        print(f"{name}: pred={r['count']} true={gt[name]['count']}")
    print("counting_error:", counting_error(pred, true))


if __name__ == "__main__":
    main()

import json
import cv2
import os
import glob
import pandas as pd

from corseacare.pipeline.contour import ContourDetector, ContourSegmenter
from corseacare.pipeline.sequential import Pipeline
from corseacare.viz import draw_overlay
from corseacare.metrics import counting_error

gt = json.load(open("data/test_run/ground_truth.json"))
pipe = Pipeline(ContourDetector(), ContourSegmenter(), mm_per_px=0.1)

os.makedirs("data/test_run/overlays", exist_ok=True)
rows = []
pred = []
true = []
per_image = []

for name in sorted(gt):
    img = cv2.imread("data/test_run/images/" + name)
    r = pipe.run(img)
    for rec in r["records"]:
        rec["image"] = name
        rows.append(rec)
    cv2.imwrite("data/test_run/overlays/" + name, draw_overlay(img, r["detections"], r["masks"]))
    pred.append(r["count"])
    true.append(gt[name]["count"])
    per_image.append({"image": name, "pred_count": r["count"], "true_count": gt[name]["count"]})

pd.DataFrame(rows).to_csv("data/test_run/counts.csv", index=False)
err = counting_error(pred, true)

print(json.dumps({
    "per_image": per_image,
    "mae": err["mae"],
    "mape": err["mape"],
    "csv": "data/test_run/counts.csv",
}, indent=2))

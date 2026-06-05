import numpy as np
import cv2

FEATURE_NAMES = ["hue", "sat", "val", "area_mm2", "aspect", "solidity", "extent"]


def masked_features(crop_bgr, mask, mm_per_px) -> dict:
    m = mask > 0
    if not m.any():
        return {n: 0.0 for n in FEATURE_NAMES}
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)[m]
    hue, sat, val = (float(np.median(hsv[:, i])) for i in range(3))
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    area_px = float(int(m.sum()))
    contour_area = float(cv2.contourArea(c)) or area_px
    x, y, w, h = cv2.boundingRect(c)
    aspect = max(w, h) / max(1, min(w, h))
    hull = cv2.contourArea(cv2.convexHull(c))
    solidity = contour_area / hull if hull > 0 else 0.0
    extent = contour_area / (w * h) if w * h > 0 else 0.0
    return {"hue": hue, "sat": sat, "val": val, "area_mm2": area_px * (mm_per_px ** 2),
            "aspect": float(aspect), "solidity": float(solidity), "extent": float(extent)}


def feature_vector(d) -> np.ndarray:
    return np.array([d[n] for n in FEATURE_NAMES], dtype=float)

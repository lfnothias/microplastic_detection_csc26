"""Sieve-circle detection + ROI gating.

The sample sits inside a circular sieve; everything outside it (rim, table, tools) is noise.
`detect_sieve_circle` finds that circle (Hough on a 500px thumbnail, central-circle fallback),
and `keep_inside_circle` drops detections whose centre falls outside it — a cheap, training-free
precision boost for tiled inference. Same detection recipe as the pre-annotator / mesh window.
"""
import cv2
import numpy as np


def detect_sieve_circle(img_bgr, fallback_frac=0.42):
    """Return (cx, cy, r) of the circular sieve in full-resolution pixels.

    Falls back to a generous central circle (the sieve is roughly centred) when Hough finds
    nothing.
    """
    h, w = img_bgr.shape[:2]
    ds = 500.0 / max(h, w)
    small = cv2.resize(img_bgr, (max(1, int(w * ds)), max(1, int(h * ds))))
    sh, sw = small.shape[:2]
    gray = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=sh, param1=100,
                               param2=40, minRadius=int(0.22 * min(sh, sw)),
                               maxRadius=int(0.50 * min(sh, sw)))
    if circles is not None:
        cx, cy, r = (np.around(circles[0][0]).astype(float) / ds)
        return float(cx), float(cy), float(r)
    return w / 2.0, h / 2.0, fallback_frac * min(h, w)


def keep_inside_circle(dets, circle, margin=0.98):
    """Filter (cls, conf, x1, y1, x2, y2) detections to those centred within margin*r."""
    cx, cy, r = circle
    rr = (r * margin) ** 2
    out = []
    for d in dets:
        x1, y1, x2, y2 = d[2], d[3], d[4], d[5]
        bx, by = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if (bx - cx) ** 2 + (by - cy) ** 2 <= rr:
            out.append(d)
    return out

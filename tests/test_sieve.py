import numpy as np
import cv2

from corseacare.sieve import detect_sieve_circle, keep_inside_circle


def test_detect_sieve_circle_synthetic():
    img = np.full((800, 800, 3), 30, np.uint8)
    cv2.circle(img, (400, 400), 300, (200, 200, 200), 8)  # bright ring on dark bg
    cx, cy, r = detect_sieve_circle(img)
    assert abs(cx - 400) < 40 and abs(cy - 400) < 40
    assert abs(r - 300) < 60


def test_detect_sieve_circle_fallback_when_none():
    img = np.full((600, 400, 3), 127, np.uint8)  # flat — no circle to find
    cx, cy, r = detect_sieve_circle(img)
    assert abs(cx - 200) < 1 and abs(cy - 300) < 1
    assert abs(r - 0.42 * 400) < 1e-6


def test_keep_inside_circle_filters_by_centre():
    circle = (100.0, 100.0, 50.0)
    dets = [
        (0, 0.9, 90, 90, 110, 110),     # centre (100,100) -> inside
        (0, 0.8, 0, 0, 10, 10),         # centre (5,5)     -> outside
        (0, 0.7, 145, 145, 155, 155),   # centre (150,150) -> outside (dist ~70)
    ]
    kept = keep_inside_circle(dets, circle, margin=1.0)
    assert len(kept) == 1 and kept[0][1] == 0.9


def test_keep_inside_circle_margin_shrinks_roi():
    circle = (100.0, 100.0, 50.0)
    dets = [(0, 0.9, 143, 95, 153, 105)]  # centre (148,100), dist 48 < 50 but > 0.9*50=45
    assert len(keep_inside_circle(dets, circle, margin=1.0)) == 1
    assert len(keep_inside_circle(dets, circle, margin=0.9)) == 0

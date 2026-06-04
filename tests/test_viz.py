import numpy as np
from corseacare.types import Detection
from corseacare.viz import draw_overlay


def test_overlay_returns_same_shape_and_changes_pixels():
    img = np.zeros((40, 40, 3), np.uint8)
    dets = [Detection((5, 5, 20, 20), 0, "fragment", 0.9)]
    masks = [np.zeros((40, 40), np.uint8)]; masks[0][5:20, 5:20] = 1
    out = draw_overlay(img, dets, masks)
    assert out.shape == img.shape
    assert out.sum() > 0           # something was drawn

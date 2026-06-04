import numpy as np
from corseacare.data.convert import bbox_from_mask, masks_to_yolo_lines


def test_bbox_from_mask():
    m = np.zeros((100, 100), np.uint8); m[10:30, 20:60] = 1
    assert bbox_from_mask(m) == (20, 10, 60, 30)


def test_masks_to_yolo_lines_normalised():
    m = np.zeros((100, 100), np.uint8); m[10:30, 20:60] = 1
    lines = masks_to_yolo_lines([m], [2], img_w=100, img_h=100)
    # class cx cy w h -> center (40,20), w=40,h=20 normalised
    assert lines[0] == "2 0.400000 0.200000 0.400000 0.200000"

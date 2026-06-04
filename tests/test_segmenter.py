import numpy as np
from corseacare.types import Detection
from corseacare.pipeline.segment_sam2 import Segmenter, FakeSegmenter


def test_fake_segmenter_returns_box_filled_masks():
    seg: Segmenter = FakeSegmenter()
    img = np.zeros((20, 20, 3), np.uint8)
    dets = [Detection((5, 5, 15, 15), 0, "fragment", 0.9)]
    masks = seg.segment(img, dets)
    assert len(masks) == 1
    assert masks[0].shape == (20, 20)
    assert masks[0][10, 10] == 1 and masks[0][0, 0] == 0

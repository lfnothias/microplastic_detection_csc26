import numpy as np
from corseacare.types import Detection
from corseacare.pipeline.segment_sam2 import FakeSegmenter
from corseacare.data.annotate_sam2 import boxes_to_yolo_for_review


def test_boxes_to_yolo_for_review_uses_segmenter_masks():
    img = np.zeros((40, 40, 3), np.uint8)
    boxes = [Detection((4, 4, 24, 24), 0, "fragment", 1.0)]
    lines = boxes_to_yolo_for_review(img, boxes, FakeSegmenter())
    assert lines[0].startswith("0 ")

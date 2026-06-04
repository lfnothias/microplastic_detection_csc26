import cv2
import numpy as np

from corseacare.pipeline.contour import ContourDetector, ContourSegmenter
from corseacare.pipeline.detector import Detector
from corseacare.pipeline.segment_sam2 import Segmenter


def _make_image():
    img = np.full((100, 100, 3), 20, np.uint8)  # dark background
    cv2.circle(img, (25, 25), 10, (255, 255, 255), -1)
    cv2.circle(img, (75, 75), 10, (255, 255, 255), -1)
    return img


def test_contour_detector_finds_two_particles():
    det = ContourDetector()
    assert isinstance(det, Detector)
    detections = det.predict(_make_image())
    assert len(detections) == 2


def test_contour_segmenter_masks_inside_bboxes():
    img = _make_image()
    detector = ContourDetector()
    segmenter = ContourSegmenter()
    assert isinstance(segmenter, Segmenter)

    detections = detector.predict(img)
    masks = segmenter.segment(img, detections)
    assert len(masks) == 2

    for d, m in zip(detections, masks):
        assert m.shape == img.shape[:2]
        assert int(m.sum()) > 0
        x1, y1, x2, y2 = (int(round(v)) for v in d.xyxy)
        # Every set pixel must fall inside this detection's bbox.
        outside = m.copy()
        outside[y1:y2, x1:x2] = 0
        assert int(outside.sum()) == 0

import numpy as np
from corseacare.types import Detection
from corseacare.pipeline.detector import Detector, FakeDetector


def test_fake_detector_returns_configured_detections():
    dets = [Detection((0, 0, 5, 5), 0, "fragment", 0.9)]
    det: Detector = FakeDetector(dets)
    out = det.predict(np.zeros((10, 10, 3), np.uint8))
    assert out == dets


def test_detector_protocol_runtime_checkable():
    assert isinstance(FakeDetector([]), Detector)

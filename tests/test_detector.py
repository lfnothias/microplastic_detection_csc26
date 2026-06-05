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


def test_tiled_detector_converts_to_detections_no_gate():
    from corseacare.pipeline.detector import TiledDetector
    img = np.zeros((40, 40, 3), np.uint8)
    det = TiledDetector(class_names=["fragment"], tile=20, overlap=0.0, roi_gate=False,
                        detect_tile=lambda t: [(0, 0.9, 5, 5, 10, 10)])
    out = det.predict(img)
    assert len(out) == 4
    assert all(isinstance(d, Detection) and d.class_name == "fragment" for d in out)
    assert isinstance(det, Detector)


def test_tiled_detector_roi_gate_drops_outside_circle():
    from corseacare.pipeline.detector import TiledDetector
    img = np.zeros((400, 400, 3), np.uint8)   # no circle -> central-circle fallback (r=168)
    det = TiledDetector(class_names=["fragment"], tile=400, overlap=0.0, roi_gate=True,
                        detect_tile=lambda t: [(0, 0.9, 195, 195, 205, 205),   # centre -> kept
                                               (0, 0.8, 5, 5, 15, 15)])        # corner -> dropped
    out = det.predict(img)
    assert len(out) == 1 and abs(out[0].confidence - 0.9) < 1e-9

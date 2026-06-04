from typing import Protocol, runtime_checkable
import numpy as np
from corseacare.types import Detection


@runtime_checkable
class Detector(Protocol):
    def predict(self, image_bgr: np.ndarray) -> list[Detection]: ...


class FakeDetector:
    """Deterministic detector for tests."""
    def __init__(self, detections: list[Detection]):
        self._detections = list(detections)

    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        return list(self._detections)


class UltralyticsDetector:
    """YOLO11 backend (AGPL-3.0). Swappable behind the Detector protocol.

    Requires the optional ML deps:
        uv pip install "ultralytics>=8.3"
    """
    def __init__(self, weights: str, class_names: list[str], conf: float = 0.25):
        from ultralytics import YOLO          # imported lazily so unit tests don't need it
        self._model = YOLO(weights)
        self._names = class_names
        self._conf = conf

    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        res = self._model.predict(image_bgr, conf=self._conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            cid = int(b.cls.item())
            name = self._names[cid] if cid < len(self._names) else str(cid)
            xyxy = tuple(float(v) for v in b.xyxy[0].tolist())
            out.append(Detection(xyxy=xyxy, class_id=cid, class_name=name, confidence=float(b.conf.item())))
        return out

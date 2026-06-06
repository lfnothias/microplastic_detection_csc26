from typing import Protocol, runtime_checkable
import numpy as np
from corseacare.types import Detection
from corseacare.tiling import tiled_detect
from corseacare.sieve import detect_sieve_circle, keep_inside_circle


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


class TiledDetector:
    """Detector that slices the image into overlapping tiles (SAHI-style) so tiny particles are
    detectable, merges per-tile YOLO boxes with NMS, and optionally gates to the sieve ROI.
    Implements the Detector protocol — a drop-in for the Pipeline that fixes full-frame collapse
    on tiny particles.

        uv pip install "ultralytics>=8.3"
    """
    def __init__(self, weights: str = "", class_names: list[str] | None = None, conf: float = 0.25,
                 tile: int = 640, overlap: float = 0.3, iou: float = 0.5, roi_gate: bool = True,
                 roi_margin: float = 0.98, detect_tile=None):
        self._names = class_names or []
        self._conf, self._tile, self._overlap, self._iou = conf, tile, overlap, iou
        self._roi_gate, self._roi_margin = roi_gate, roi_margin
        if detect_tile is not None:
            self._detect_tile = detect_tile            # injected (tests / custom backends)
        else:
            from ultralytics import YOLO               # lazy: unit tests don't need it
            self._model = YOLO(weights)
            self._detect_tile = self._yolo_tile

    def _yolo_tile(self, tile_bgr):
        res = self._model.predict(tile_bgr, conf=self._conf, verbose=False)[0]
        return [(int(b.cls.item()), float(b.conf.item()),
                 *[float(v) for v in b.xyxy[0].tolist()]) for b in res.boxes]

    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        merged = tiled_detect(image_bgr, self._detect_tile, self._tile, self._overlap, self._iou)
        if self._roi_gate:
            merged = keep_inside_circle(merged, detect_sieve_circle(image_bgr), margin=self._roi_margin)
        out = []
        for (cls, conf, x1, y1, x2, y2) in merged:
            name = self._names[cls] if 0 <= cls < len(self._names) else str(cls)
            out.append(Detection(xyxy=(x1, y1, x2, y2), class_id=cls, class_name=name, confidence=conf))
        return out

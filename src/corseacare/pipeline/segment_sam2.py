from typing import Protocol, runtime_checkable
import numpy as np
from corseacare.types import Detection


@runtime_checkable
class Segmenter(Protocol):
    def segment(self, image_bgr: np.ndarray, detections: list[Detection]) -> list[np.ndarray]: ...


class FakeSegmenter:
    """Fills each detection box as its mask. Deterministic, for tests."""
    def segment(self, image_bgr, detections):
        h, w = image_bgr.shape[:2]
        masks = []
        for d in detections:
            m = np.zeros((h, w), np.uint8)
            x1, y1, x2, y2 = (int(round(v)) for v in d.xyxy)
            m[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
            masks.append(m)
        return masks


class SAM2Segmenter:
    """Box-prompted SAM2 (Apache-2.0). Uses a small/tiny checkpoint for local use.

    Requires the optional ML deps:
        uv pip install "sam-2 @ git+https://github.com/facebookresearch/sam2.git"
    """
    def __init__(self, checkpoint: str, model_cfg: str, device: str = "cpu"):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self._predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))

    def segment(self, image_bgr, detections):
        import cv2
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(rgb)
        masks = []
        for d in detections:
            box = np.array(d.xyxy, dtype=np.float32)
            m, _, _ = self._predictor.predict(box=box[None, :], multimask_output=False)
            masks.append((m[0] > 0).astype(np.uint8))
        return masks

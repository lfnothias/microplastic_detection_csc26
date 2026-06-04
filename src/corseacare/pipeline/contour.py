"""Torch-free classical detector + segmenter.

These implementations rely only on OpenCV/NumPy so the pipeline can run fully
offline (no torch / ultralytics / sam2 required). Particles are assumed to be
brighter than a dark background.
"""
import cv2
import numpy as np

from corseacare.types import Detection


class ContourDetector:
    """Threshold + connected-component detector satisfying the Detector protocol."""

    def __init__(self, class_name: str = "fragment", thresh: int = 60, min_area: float = 30):
        self.class_name = class_name
        self.thresh = thresh
        self.min_area = min_area

    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Particles are BRIGHTER than the dark background.
        _, binary = cv2.threshold(gray, self.thresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for c in contours:
            if cv2.contourArea(c) >= self.min_area:
                x, y, w, h = cv2.boundingRect(c)
                detections.append(
                    Detection(
                        xyxy=(float(x), float(y), float(x + w), float(y + h)),
                        class_id=0,
                        class_name=self.class_name,
                        confidence=1.0,
                    )
                )
        return detections


class ContourSegmenter:
    """Threshold-based segmenter satisfying the Segmenter protocol.

    For each detection, the returned mask is set (1) where the image is brighter
    than ``thresh`` AND the pixel falls inside the detection bounding box.
    """

    def __init__(self, thresh: int = 60):
        self.thresh = thresh

    def segment(self, image_bgr: np.ndarray, detections: list[Detection]) -> list[np.ndarray]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        binary = (gray > self.thresh).astype(np.uint8)
        h, w = binary.shape[:2]
        masks: list[np.ndarray] = []
        for d in detections:
            x1, y1, x2, y2 = (int(round(v)) for v in d.xyxy)
            x1 = max(0, min(w, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1))
            y2 = max(0, min(h, y2))
            mask = np.zeros((h, w), np.uint8)
            mask[y1:y2, x1:x2] = binary[y1:y2, x1:x2]
            masks.append(mask)
        return masks

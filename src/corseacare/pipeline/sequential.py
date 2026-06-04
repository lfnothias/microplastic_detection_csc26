import numpy as np
from corseacare.types import SampleMetadata
from corseacare.pipeline.detector import Detector
from corseacare.pipeline.segment_sam2 import Segmenter
from corseacare.measure import assemble_records


class Pipeline:
    def __init__(self, detector: Detector, segmenter: Segmenter, mm_per_px: float):
        self.detector = detector
        self.segmenter = segmenter
        self.mm_per_px = mm_per_px

    def run(self, image_bgr: np.ndarray, sample: SampleMetadata | None = None) -> dict:
        detections = self.detector.predict(image_bgr)
        masks = self.segmenter.segment(image_bgr, detections)
        records = assemble_records(image_bgr, detections, masks, self.mm_per_px, sample)
        return {"count": len(detections), "detections": detections, "masks": masks, "records": records}

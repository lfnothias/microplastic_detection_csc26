import numpy as np
from corseacare.types import Detection, SampleMetadata
from corseacare.pipeline.detector import FakeDetector
from corseacare.pipeline.segment_sam2 import FakeSegmenter
from corseacare.pipeline.sequential import Pipeline


def test_pipeline_counts_and_measures():
    img = np.zeros((20, 20, 3), np.uint8); img[:, :, 2] = 255
    dets = [Detection((5, 5, 15, 15), 0, "fragment", 0.9)]
    pipe = Pipeline(FakeDetector(dets), FakeSegmenter(), mm_per_px=0.5)
    result = pipe.run(img, sample=SampleMetadata(location="Lisula"))
    assert result["count"] == 1
    assert result["records"][0]["colour"] == "rouge"
    assert result["records"][0]["location"] == "Lisula"
    assert len(result["masks"]) == 1

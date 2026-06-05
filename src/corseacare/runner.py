import os
from corseacare.config import Config
from corseacare.pipeline.sequential import Pipeline
from corseacare.pipeline.detector import FakeDetector, UltralyticsDetector, TiledDetector
from corseacare.pipeline.segment_sam2 import FakeSegmenter, SAM2Segmenter
from corseacare.types import Detection


def build_pipeline(cfg: Config, mm_per_px: float, tiled: bool = False) -> Pipeline:
    """Build the pipeline. Set CORSEACARE_FAKE=1 to use deterministic Fake backends
    (no torch / models needed) — used by tests and offline smoke runs.

    `tiled=True` uses the TiledDetector (SAHI-style tiling + sieve-ROI gating), which is what
    actually detects the tiny particles; the default full-frame detector under-detects them.
    """
    if os.environ.get("CORSEACARE_FAKE") == "1":
        det = FakeDetector([Detection((5, 5, 15, 15), 0, cfg.classes[0], 0.99)])
        seg = FakeSegmenter()
    else:
        if tiled:
            det = TiledDetector(cfg.weights, cfg.classes, cfg.conf_threshold,
                                cfg.tile_size, cfg.tile_overlap, roi_gate=cfg.roi_gate)
        else:
            det = UltralyticsDetector(cfg.weights, cfg.classes, cfg.conf_threshold)
        seg = SAM2Segmenter(cfg.sam2_checkpoint, cfg.sam2_model_cfg)
    return Pipeline(det, seg, mm_per_px)

import numpy as np
import cv2
from skimage.measure import label, regionprops

from corseacare.types import Detection, ParticleRecord, SampleMetadata


# --- colour -----------------------------------------------------------------

def classify_colour(h: float, s: float, v: float) -> str:
    """h in [0,179], s,v in [0,255] (OpenCV HSV)."""
    if v < 50:
        return "noir"
    if s < 40:
        return "blanc/transparent" if v > 180 else "gris"
    if h < 10 or h >= 170:
        return "rouge"
    if h < 22:
        return "orange"
    if h < 34:
        return "jaune"
    if h < 85:
        return "vert"
    if h < 130:
        return "bleu"
    if h < 160:
        return "violet"
    return "rouge"


def particle_colour(image_bgr: np.ndarray, mask: np.ndarray) -> dict:
    m = mask.astype(bool)
    if not m.any():
        return {"colour": "inconnu", "hue": 0.0, "sat": 0.0, "val": 0.0, "mean_rgb": (0.0, 0.0, 0.0)}
    bgr = image_bgr[m].astype(np.float64)
    mean_rgb = (float(bgr[:, 2].mean()), float(bgr[:, 1].mean()), float(bgr[:, 0].mean()))
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[m]
    h = float(np.median(hsv[:, 0])); s = float(np.median(hsv[:, 1])); v = float(np.median(hsv[:, 2]))
    return {"colour": classify_colour(h, s, v), "hue": h, "sat": s, "val": v, "mean_rgb": mean_rgb}


# --- size -------------------------------------------------------------------

def particle_size(mask: np.ndarray, mm_per_px: float) -> dict:
    m = (mask > 0).astype(np.uint8)
    area_px = int(m.sum())
    if area_px == 0:
        return {"area_px": 0, "area_mm2": 0.0, "max_feret_mm": 0.0}
    props = regionprops(label(m))
    biggest = max(props, key=lambda r: r.area)
    try:
        feret_px = float(biggest.feret_diameter_max)
    except (AttributeError, ValueError):
        feret_px = float(max(biggest.bbox[2] - biggest.bbox[0], biggest.bbox[3] - biggest.bbox[1]))
    return {
        "area_px": area_px,
        "area_mm2": area_px * (mm_per_px ** 2),
        "max_feret_mm": feret_px * mm_per_px,
    }


# --- records ----------------------------------------------------------------

def measure_particle(image_bgr, mask, detection: Detection, mm_per_px: float) -> ParticleRecord:
    colour = particle_colour(image_bgr, mask)
    size = particle_size(mask, mm_per_px)
    return ParticleRecord(
        class_name=detection.class_name,
        confidence=detection.confidence,
        colour=colour["colour"],
        area_mm2=size["area_mm2"],
        max_feret_mm=size["max_feret_mm"],
        area_px=size["area_px"],
        xyxy=detection.xyxy,
        extra={"hue": colour["hue"], "sat": colour["sat"], "val": colour["val"]},
    )


def assemble_records(image_bgr, detections, masks, mm_per_px, sample: SampleMetadata | None = None):
    rows = []
    for det, mask in zip(detections, masks):
        rec = measure_particle(image_bgr, mask, det, mm_per_px)
        rows.append(rec.to_row(sample))
    return rows

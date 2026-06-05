"""Aggregate per-particle records into a per-sample report.

Consumes the row dicts produced by `measure.assemble_records` (keys: `class_name`, `colour`,
`area_mm2`, `max_feret_mm`, ...) and produces counts by class & colour, size statistics and a
size-class histogram, projected-area totals, and a clearly-labelled volume estimate. If a tow
volume is supplied, also a concentration (particles / m³).

Honesty note: a 2-D photo cannot measure true volume or mass. `estimate_volume_mm3` returns an
ESTIMATE under a stated shape assumption — the robust quantity is projected **area (mm²)**.
"""
import statistics
from collections import Counter

# size classes by max Feret (mm) — aligned with the 1 mm / 2 mm sieve fractions
SIZE_EDGES_MM = [0.0, 1.0, 2.0, 5.0, 10.0, float("inf")]
SIZE_LABELS = ["<1mm", "1-2mm", "2-5mm", "5-10mm", ">10mm"]


def _stats(values):
    vals = [v for v in values if v and v > 0]
    if not vals:
        return {"n": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {"n": len(vals), "mean": statistics.fmean(vals), "median": statistics.median(vals),
            "min": min(vals), "max": max(vals)}


def size_histogram(feret_mm_values, edges=SIZE_EDGES_MM, labels=SIZE_LABELS):
    """Count particles into size classes by max Feret (mm). Zero/None sizes are skipped."""
    hist = {lab: 0 for lab in labels}
    for v in feret_mm_values:
        if not v or v <= 0:
            continue
        for i in range(len(labels)):
            if edges[i] <= v < edges[i + 1]:
                hist[labels[i]] += 1
                break
    return hist


def estimate_volume_mm3(areas_mm2, model="area_power", thickness_mm=0.5):
    """ROUGH volume proxy from 2-D projected areas — NOT a measurement.

    - 'area_power':  V = Σ area**1.5  (shape-free; assumes depth ∝ √area)
    - 'thickness':   V = Σ area * thickness_mm  (flat-flake assumption)
    """
    a = [x for x in areas_mm2 if x and x > 0]
    if model == "thickness":
        total = sum(x * thickness_mm for x in a)
        assumption = f"flat flakes, thickness={thickness_mm}mm"
    else:
        total = sum(x ** 1.5 for x in a)
        assumption = "depth proportional to sqrt(area) (V = area^1.5)"
    return {"total_mm3": total, "model": model, "assumption": assumption,
            "caveat": "A 2-D photo cannot measure true volume; this is an estimate."}


def summarize(records, tow_volume_m3=None, volume_model="area_power", thickness_mm=0.5):
    """Aggregate one sample's particle records (list of row dicts) into a report dict."""
    n = len(records)
    by_class = Counter(r.get("class_name", "?") for r in records)
    by_colour = Counter(r.get("colour", "?") for r in records)
    ferets = [r.get("max_feret_mm", 0.0) for r in records]
    areas = [r.get("area_mm2", 0.0) for r in records]

    area_by_class = {}
    for r in records:
        cls = r.get("class_name", "?")
        area_by_class[cls] = area_by_class.get(cls, 0.0) + (r.get("area_mm2", 0.0) or 0.0)

    report = {
        "n_particles": n,
        "counts_by_class": dict(by_class),
        "counts_by_colour": dict(by_colour),
        "size_mm": _stats(ferets),
        "size_histogram_mm": size_histogram(ferets),
        "area_mm2": {"total": sum(x for x in areas if x and x > 0), "by_class": area_by_class},
        "volume_estimate": estimate_volume_mm3(areas, volume_model, thickness_mm),
    }
    if tow_volume_m3 and tow_volume_m3 > 0:
        report["concentration_per_m3"] = n / tow_volume_m3
    return report

"""Mesh-as-ruler calibration: the sieve's regular wire grid is a periodic scale.

`detect_mesh_period` recovers the wire pitch in pixels (px/cell) from the dominant
non-DC peak of the 2-D FFT. It uses the *radial* spatial frequency (not axis-aligned
row/column profiles), so it is robust to a few degrees of sieve rotation and to partial
occlusion by particles (the FFT averages over the whole window).

The detected period is the **pitch** (wire-to-wire) = aperture + wire_diameter, while a
"1 mm / 2 mm sieve" denotes the **aperture**. `solve_pitch_mm` back-solves the true pitch
in mm from a one-time ruler-derived px/mm; `px_per_mm_from_period` then turns any photo's
detected period into a per-photo scale.
"""
import numpy as np


def detect_mesh_period(gray, min_px=5.0, max_px=80.0):
    """Return (period_px, confidence) for the dominant grid period in `gray`.

    Only periods in [min_px, max_px] are considered. `confidence` is the peak magnitude
    divided by the median magnitude in that band — values >~3 indicate a real periodic
    signal; a flat/aperiodic image returns ~1.
    """
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError("gray must be a 2-D array")
    H, W = g.shape
    g = g - g.mean()
    win = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    F = np.fft.fftshift(np.abs(np.fft.fft2(g * win)))

    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    fx = (xx - cx) / W                      # cycles / px (x)
    fy = (yy - cy) / H                      # cycles / px (y)
    freq = np.sqrt(fx ** 2 + fy ** 2)
    with np.errstate(divide="ignore"):
        period = np.where(freq > 0, 1.0 / freq, np.inf)

    band = (period >= min_px) & (period <= max_px)
    if not band.any():
        return 0.0, 0.0
    mags = F[band]
    idx = int(np.argmax(F * band))
    peak_period = float(period.flat[idx])
    med = float(np.median(mags)) or 1e-9
    return peak_period, float(F.flat[idx] / med)


def solve_pitch_mm(period_px, px_per_mm):
    """True mesh pitch in mm from an observed period (px) and a ruler-derived px/mm."""
    if px_per_mm <= 0:
        raise ValueError("px_per_mm must be > 0")
    return period_px / px_per_mm


def px_per_mm_from_period(period_px, pitch_mm):
    """Per-photo scale (px/mm) from an observed period (px) and the known pitch (mm)."""
    if pitch_mm <= 0:
        raise ValueError("pitch_mm must be > 0")
    return period_px / pitch_mm

import numpy as np

from corseacare.calib import detect_mesh_period, solve_pitch_mm, px_per_mm_from_period


def _grid(n, period, theta_deg=0.0):
    x = np.arange(n)
    X, Y = np.meshgrid(x, x)
    th = np.deg2rad(theta_deg)
    Xr = X * np.cos(th) + Y * np.sin(th)
    Yr = -X * np.sin(th) + Y * np.cos(th)
    g = np.sin(2 * np.pi * Xr / period) + np.sin(2 * np.pi * Yr / period)
    return ((g - g.min()) / (g.max() - g.min()) * 255).astype(np.uint8)


def test_detect_mesh_period_axis_aligned():
    p, conf = detect_mesh_period(_grid(256, 16), min_px=5, max_px=80)
    assert abs(p - 16) < 1.5
    assert conf > 3


def test_detect_mesh_period_rotated():
    # radial-frequency peak is rotation-robust
    p, conf = detect_mesh_period(_grid(256, 20, theta_deg=8), min_px=5, max_px=80)
    assert abs(p - 20) < 2.0


def test_detect_mesh_period_no_signal_returns_low_conf():
    flat = np.full((128, 128), 127, dtype=np.uint8)
    _, conf = detect_mesh_period(flat, min_px=5, max_px=60)
    assert conf < 3


def test_load_mm_per_px_map(tmp_path):
    from corseacare.calib import load_mm_per_px_map
    m = tmp_path / "samples.csv"
    m.write_text("image,sample_id,px_per_mm\n"
                 "a.jpg,S1,20\n"        # 20 px/mm -> 0.05 mm/px
                 "b.jpg,S1,\n"          # blank -> skipped
                 "c.jpg,S2,0\n")        # zero -> skipped
    out = load_mm_per_px_map(m)
    assert set(out) == {"a.jpg"}
    assert abs(out["a.jpg"] - 0.05) < 1e-9


def test_load_mm_per_px_map_missing_file(tmp_path):
    from corseacare.calib import load_mm_per_px_map
    assert load_mm_per_px_map(tmp_path / "nope.csv") == {}


def test_solve_pitch_mm():
    # 40 px period observed at 20 px/mm => true pitch 2.0 mm
    assert abs(solve_pitch_mm(40.0, 20.0) - 2.0) < 1e-9


def test_px_per_mm_from_period_is_inverse():
    assert abs(px_per_mm_from_period(40.0, 2.0) - 20.0) < 1e-9


def test_solve_pitch_rejects_bad_scale():
    import pytest
    with pytest.raises(ValueError):
        solve_pitch_mm(40.0, 0.0)

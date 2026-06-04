import numpy as np
from corseacare.measure import particle_size


def test_square_area_scales_with_mm_per_px():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1            # 10x10 = 100 px
    out = particle_size(mask, mm_per_px=0.5)
    assert out["area_px"] == 100
    assert out["area_mm2"] == 25.0  # 100 * 0.5^2
    assert out["max_feret_mm"] > 0


def test_empty_mask_is_zero():
    out = particle_size(np.zeros((5, 5), np.uint8), mm_per_px=1.0)
    assert out["area_px"] == 0 and out["area_mm2"] == 0.0

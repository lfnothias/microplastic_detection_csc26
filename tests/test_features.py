import numpy as np
from corseacare.features import masked_features, feature_vector, FEATURE_NAMES


def test_red_square_features():
    img = np.zeros((20, 20, 3), np.uint8); img[:, :, 2] = 255   # red
    mask = np.zeros((20, 20), np.uint8); mask[5:15, 5:15] = 1   # 10x10
    f = masked_features(img, mask, mm_per_px=0.5)
    assert round(f["area_mm2"], 2) == 25.0          # 100 px * 0.5^2
    assert 0.9 <= f["aspect"] <= 1.2                 # square
    assert f["solidity"] > 0.9
    assert f["hue"] <= 10 or f["hue"] >= 170         # red hue
    assert feature_vector(f).shape == (len(FEATURE_NAMES),)


def test_empty_mask_is_zeros():
    f = masked_features(np.zeros((5, 5, 3), np.uint8), np.zeros((5, 5), np.uint8), 1.0)
    assert feature_vector(f).tolist() == [0.0] * len(FEATURE_NAMES)

import numpy as np
from corseacare.measure import classify_colour, particle_colour


def test_classify_white_low_sat_high_val():
    assert classify_colour(h=0, s=10, v=240) == "blanc/transparent"


def test_classify_blue_hue():
    assert classify_colour(h=110, s=200, v=200) == "bleu"


def test_particle_colour_pure_red_patch():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :, 2] = 255  # BGR -> red
    mask = np.ones((10, 10), dtype=np.uint8)
    out = particle_colour(img, mask)
    assert out["colour"] == "rouge"
    assert round(out["mean_rgb"][0]) == 255  # R channel

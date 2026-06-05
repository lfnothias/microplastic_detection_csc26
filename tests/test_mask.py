import numpy as np
from corseacare.mask import particle_mask


def test_colored_square_on_grey():
    img = np.full((40, 40, 3), 128, np.uint8)
    img[10:30, 10:30] = (0, 0, 255)            # red (saturated) square
    m = particle_mask(img)
    assert m[20, 20] == 1 and m[2, 2] == 0


def test_dark_grey_square_by_contrast():
    img = np.full((40, 40, 3), 130, np.uint8)
    img[15:25, 15:25] = 30                     # low-saturation but high contrast
    m = particle_mask(img)
    assert m[20, 20] == 1 and m[2, 2] == 0

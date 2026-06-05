import numpy as np
import cv2


def particle_mask(crop_bgr, sat_thresh=60, gray_delta=40):
    """Isolate the particle from the mesh inside a tight crop.

    Foreground = saturated pixels OR pixels whose grey value differs strongly from the
    border (background reference). Returns uint8 HxW mask in {0,1}.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg = float(np.median(border))
    fg = ((hsv[:, :, 1] > sat_thresh) | (np.abs(gray - bg) > gray_delta)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return fg

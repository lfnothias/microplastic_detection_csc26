import numpy as np


def test_run_on_image_headless(monkeypatch):
    monkeypatch.setenv("CORSEACARE_FAKE", "1")
    from app.server import run_on_image
    img = np.zeros((30, 30, 3), np.uint8); img[:, :, 2] = 255
    count, df, overlay = run_on_image(img, mm_per_px=0.5)
    assert count == 1
    assert "area_mm2" in df.columns
    assert overlay.shape == img.shape

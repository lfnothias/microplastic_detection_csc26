import numpy as np
import cv2
from typer.testing import CliRunner
from corseacare.cli import app


def test_count_cli_writes_csv(tmp_path, monkeypatch):
    # force the Fake backends so the CLI test needs no models
    monkeypatch.setenv("CORSEACARE_FAKE", "1")
    img = np.zeros((30, 30, 3), np.uint8); img[:, :, 2] = 255
    cv2.rectangle(img, (5, 5), (15, 15), (0, 0, 255), -1)
    d = tmp_path / "imgs"; d.mkdir(); cv2.imwrite(str(d / "a.png"), img)
    out_csv = tmp_path / "out.csv"
    res = CliRunner().invoke(app, ["count", str(d), "--out", str(out_csv), "--mm-per-px", "0.5"])
    assert res.exit_code == 0, res.output
    assert out_csv.exists()
    assert "area_mm2" in out_csv.read_text()

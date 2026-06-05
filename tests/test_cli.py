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


def test_report_cli_groups_and_dedups_views(tmp_path):
    import json
    import pandas as pd
    parts = tmp_path / "counts.csv"
    pd.DataFrame([
        {"image": "v1.jpg", "class_name": "fragment", "colour": "bleu", "area_mm2": 4.0, "max_feret_mm": 2.0},
        {"image": "v1.jpg", "class_name": "fragment", "colour": "bleu", "area_mm2": 4.0, "max_feret_mm": 2.0},
        {"image": "v1.jpg", "class_name": "pellet", "colour": "gris", "area_mm2": 1.0, "max_feret_mm": 1.0},
        {"image": "v2.jpg", "class_name": "fragment", "colour": "bleu", "area_mm2": 4.0, "max_feret_mm": 2.0},
        {"image": "solo.jpg", "class_name": "film", "colour": "vert", "area_mm2": 2.0, "max_feret_mm": 1.5},
    ]).to_csv(parts, index=False)
    man = tmp_path / "samples.csv"
    pd.DataFrame([
        {"image": "v1.jpg", "sample_id": "S1"},
        {"image": "v2.jpg", "sample_id": "S1"},
        {"image": "solo.jpg", "sample_id": "S2"},
    ]).to_csv(man, index=False)
    oj, oc = tmp_path / "r.json", tmp_path / "r.csv"
    res = CliRunner().invoke(app, ["report", "--particles", str(parts), "--manifest", str(man),
                                   "--out-json", str(oj), "--out-csv", str(oc)])
    assert res.exit_code == 0, res.output
    data = json.loads(oj.read_text())
    # S1 = 2 reshuffled views -> NOT summed; representative (median-count) view v1 has 3 particles
    assert data["S1"]["n_particles"] == 3
    assert "not summed" in data["S1"]["note"]
    assert data["S2"]["n_particles"] == 1

from corseacare.report import summarize, size_histogram, estimate_volume_mm3


def recs():
    return [
        {"class_name": "fragment", "colour": "bleu", "area_mm2": 4.0, "max_feret_mm": 2.5},
        {"class_name": "fragment", "colour": "rouge", "area_mm2": 1.0, "max_feret_mm": 1.2},
        {"class_name": "pellet", "colour": "blanc/transparent", "area_mm2": 9.0, "max_feret_mm": 3.4},
        {"class_name": "autre", "colour": "vert", "area_mm2": 0.0, "max_feret_mm": 0.0},  # unmeasured
    ]


def test_counts_by_class_and_colour():
    r = summarize(recs())
    assert r["n_particles"] == 4
    assert r["counts_by_class"] == {"fragment": 2, "pellet": 1, "autre": 1}
    assert r["counts_by_colour"]["bleu"] == 1


def test_size_stats_ignore_unmeasured():
    r = summarize(recs())["size_mm"]
    assert r["n"] == 3                          # the 0.0 Feret is excluded
    assert abs(r["max"] - 3.4) < 1e-9
    assert abs(r["median"] - 2.5) < 1e-9


def test_size_histogram_bins():
    h = size_histogram([0.5, 1.2, 2.5, 7.0, 12.0, 0.0])
    assert h == {"<1mm": 1, "1-2mm": 1, "2-5mm": 1, "5-10mm": 1, ">10mm": 1}


def test_area_total_and_by_class():
    r = summarize(recs())["area_mm2"]
    assert abs(r["total"] - 14.0) < 1e-9
    assert abs(r["by_class"]["fragment"] - 5.0) < 1e-9


def test_volume_estimate_two_models():
    v_thick = estimate_volume_mm3([4.0], model="thickness", thickness_mm=0.5)
    assert abs(v_thick["total_mm3"] - 2.0) < 1e-9          # 4 * 0.5
    v_pow = estimate_volume_mm3([4.0], model="area_power")
    assert abs(v_pow["total_mm3"] - 8.0) < 1e-9            # 4 ** 1.5
    assert "estimate" in v_pow["caveat"].lower()           # honesty caveat present


def test_concentration_only_with_tow_volume():
    assert "concentration_per_m3" not in summarize(recs())
    r = summarize(recs(), tow_volume_m3=2.0)
    assert abs(r["concentration_per_m3"] - 2.0) < 1e-9     # 4 particles / 2 m3

from corseacare.config import load_config, Config


def test_default_classes_include_organic():
    cfg = Config()
    assert "matiere_organique" in cfg.classes
    assert cfg.classes[0] == "fragment"


def test_load_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("mm_per_px: 0.25\nweights: my.pt\n")
    cfg = load_config(p)
    assert cfg.mm_per_px == 0.25
    assert cfg.weights == "my.pt"

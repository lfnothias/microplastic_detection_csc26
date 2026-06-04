import numpy as np
import cv2
import yaml
from corseacare.data.prepare_deepparticle import build_yolo_dataset


def test_build_yolo_dataset_creates_split_and_yaml(tmp_path):
    # synthetic source: one image + one single-class mask
    src = tmp_path / "src"; (src / "images").mkdir(parents=True); (src / "masks").mkdir()
    img = np.zeros((50, 50, 3), np.uint8); cv2.imwrite(str(src / "images" / "a.png"), img)
    m = np.zeros((50, 50), np.uint8); m[10:20, 10:30] = 255
    cv2.imwrite(str(src / "masks" / "a.png"), m)
    out = tmp_path / "out"
    build_yolo_dataset(src, out, class_id=0, classes=["fragment"], val_frac=0.0)
    assert (out / "images" / "train" / "a.png").exists()
    assert (out / "labels" / "train" / "a.txt").read_text().startswith("0 ")
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == ["fragment"]

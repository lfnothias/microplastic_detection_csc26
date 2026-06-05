import numpy as np, cv2
from scripts.build_reference_guide import build_guide


def test_build_guide_makes_montage(tmp_path):
    cls = tmp_path / "examples" / "fragment"; cls.mkdir(parents=True)
    for i in range(2):
        cv2.imwrite(str(cls / f"{i}.png"), np.full((30, 30, 3), 200, np.uint8))
    out = tmp_path / "guide.png"
    build_guide(tmp_path / "examples", out)
    img = cv2.imread(str(out))
    assert img is not None and img.shape[0] > 0

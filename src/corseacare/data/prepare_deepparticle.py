from pathlib import Path
import shutil
import cv2
import yaml

from corseacare.data.convert import masks_to_yolo_lines


def build_yolo_dataset(src: Path, out: Path, class_id: int, classes: list[str], val_frac: float = 0.2):
    """Convert a {images/, masks/} source layout into a YOLO-det dataset on disk.

    Adapt the source-layout assumptions to the real DeepParticle structure once it is
    downloaded and its license confirmed (see plan Task 0).
    """
    src, out = Path(src), Path(out)
    images = sorted((src / "images").glob("*.png")) + sorted((src / "images").glob("*.jpg"))
    n_val = int(len(images) * val_frac)
    for i, img_path in enumerate(images):
        split = "val" if i < n_val else "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy(img_path, out / "images" / split / img_path.name)
        mask_path = src / "masks" / img_path.name
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        lines = []
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            lines = masks_to_yolo_lines([mask], [class_id], w, h)
        (out / "labels" / split / f"{img_path.stem}.txt").write_text("\n".join(lines))
    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out), "train": "images/train", "val": "images/val",
        "names": classes,
    }))

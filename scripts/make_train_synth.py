"""Generate a larger synthetic TRAIN dataset directly in YOLO-det format, so a usable
detector can be fine-tuned locally on the M4. Same visual style as the held-out test set
(data/test_run): dark-grey background with bright coloured circles/rectangles, including
blue. Single class: fragment (0). Deterministic (fixed seed).

Output: data/train_synth_yolo/{images,labels}/{train,val} + data.yaml  (gitignored via /data/).
"""
import random
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "train_synth_yolo"
SIZE = 480
COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (0, 200, 0), (255, 255, 255)]  # BGR incl. blue
N_TRAIN, N_VAL = 200, 40


def _overlaps(a, b, m=3):
    return not (a[2] + m < b[0] or b[2] + m < a[0] or a[3] + m < b[1] or b[3] + m < a[1])


def gen_one(rng):
    img = np.full((SIZE, SIZE, 3), 30, np.uint8)
    placed, labels = [], []
    for _ in range(rng.randint(3, 10)):
        for _ in range(60):
            r = rng.randint(8, 30)
            cx, cy = rng.randint(r + 2, SIZE - r - 2), rng.randint(r + 2, SIZE - r - 2)
            box = (cx - r, cy - r, cx + r, cy + r)
            if all(not _overlaps(box, p) for p in placed):
                break
        else:
            continue
        placed.append(box)
        color = rng.choice(COLORS)
        if rng.random() < 0.5:
            cv2.circle(img, (cx, cy), r, color, -1)
        else:
            cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, -1)
        x1, y1, x2, y2 = box
        labels.append(f"0 {((x1 + x2) / 2) / SIZE:.6f} {((y1 + y2) / 2) / SIZE:.6f} "
                      f"{(x2 - x1) / SIZE:.6f} {(y2 - y1) / SIZE:.6f}")
    return img, labels


def main():
    rng = random.Random(100)
    for split, n in (("train", N_TRAIN), ("val", N_VAL)):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, labels = gen_one(rng)
            cv2.imwrite(str(OUT / "images" / split / f"{i:04d}.png"), img)
            (OUT / "labels" / split / f"{i:04d}.txt").write_text("\n".join(labels))
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnames: [fragment]\n"
    )
    print(f"wrote synthetic YOLO dataset to {OUT} ({N_TRAIN} train, {N_VAL} val)")


if __name__ == "__main__":
    main()

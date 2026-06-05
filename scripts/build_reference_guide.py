"""Assemble a visual annotation guide / few-shot reference card from expert-picked example
crops, organised as <examples>/<class>/*.png. One labelled row per class.

    .venv/bin/python scripts/build_reference_guide.py <examples_dir> <out.png>
"""
import sys
from pathlib import Path
import numpy as np
import cv2

CELL = 96


def build_guide(examples_dir, out_path):
    examples_dir = Path(examples_dir)
    classes = sorted(d for d in examples_dir.iterdir() if d.is_dir())
    rows = []
    for d in classes:
        imgs = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")))[:6]
        cells = [np.full((CELL, CELL, 3), 30, np.uint8)]  # label cell
        cv2.putText(cells[0], d.name[:10], (2, CELL // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        for p in imgs:
            cells.append(cv2.resize(cv2.imread(str(p)), (CELL, CELL)))
        row = np.hstack(cells + [np.full((CELL, CELL, 3), 30, np.uint8)] * (7 - len(cells)))
        rows.append(row)
    guide = np.vstack(rows) if rows else np.zeros((CELL, CELL, 3), np.uint8)
    cv2.imwrite(str(out_path), guide)
    return out_path


if __name__ == "__main__":
    build_guide(sys.argv[1], sys.argv[2])
    print(f"guide -> {sys.argv[2]}")

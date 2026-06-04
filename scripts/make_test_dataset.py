"""Synthetic test-dataset generator for CorSeaCare.

Generates a small, fully-deterministic set of RGB images containing bright,
non-overlapping particles of known colour and size on a dark-grey background,
together with a ground-truth JSON file. Useful as a fixture for exercising the
detection / segmentation / measurement pipeline without any trained weights.

Run (from the repo root):
    .venv/bin/python scripts/make_test_dataset.py

Outputs (all under data/test_run/, which is gitignored):
    data/test_run/images/img_{i}.png
    data/test_run/ground_truth.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np

# --- configuration ----------------------------------------------------------

SEED = 7
N_IMAGES = 5
IMG_SIZE = 480  # square, in pixels
BG_BGR = (30, 30, 30)  # dark grey

# Colours given as BGR (OpenCV native), keyed by their French name.
COLOURS_BGR: dict[str, tuple[int, int, int]] = {
    "rouge": (0, 0, 255),
    "bleu": (255, 0, 0),
    "jaune": (0, 255, 255),
    "vert": (0, 200, 0),
    "blanc": (255, 255, 255),
}

MIN_PARTICLES = 3
MAX_PARTICLES = 8
MIN_HALF = 8   # min radius / half-size in px
MAX_HALF = 30  # max radius / half-size in px
MARGIN = 4     # keep particles a few px away from the image border
GAP = 2        # min gap (px) between particle bounding boxes -> non-overlap

# Output locations, anchored to the repo root (parent of scripts/).
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "test_run"
IMG_DIR = OUT_DIR / "images"
GT_PATH = OUT_DIR / "ground_truth.json"


def _bbox_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    """True if axis-aligned bboxes a, b are closer than `gap` px (i.e. overlap)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (
        ax2 + gap <= bx1
        or bx2 + gap <= ax1
        or ay2 + gap <= by1
        or by2 + gap <= ay1
    )


def _draw_particle(img: np.ndarray, shape: str, cx: int, cy: int, half: int,
                   colour_bgr: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """Draw one filled particle in-place and return its xyxy bbox (ints)."""
    if shape == "circle":
        cv2.circle(img, (cx, cy), half, colour_bgr, thickness=-1)
    else:  # rectangle
        cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half),
                      colour_bgr, thickness=-1)
    return (cx - half, cy - half, cx + half, cy + half)


def _measured_area_px(img: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    """Count rendered foreground (non-background) pixels inside `bbox`.

    Matches how corseacare.measure.particle_size derives area_px from a mask,
    so ground truth lines up with what the pipeline would measure.
    """
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2 + 1, x1:x2 + 1]
    bg = np.array(BG_BGR, dtype=img.dtype)
    fg = np.any(crop != bg, axis=2)
    return int(fg.sum())


def generate() -> dict:
    random.seed(SEED)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    colour_names = list(COLOURS_BGR.keys())
    ground_truth: dict[str, dict] = {}

    for i in range(N_IMAGES):
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        img[:] = BG_BGR

        n = random.randint(MIN_PARTICLES, MAX_PARTICLES)
        placed_bboxes: list[tuple[int, int, int, int]] = []
        particles: list[dict] = []

        for _ in range(n):
            # Try to place a non-overlapping particle; give up after many tries.
            for _attempt in range(500):
                half = random.randint(MIN_HALF, MAX_HALF)
                lo = MARGIN + half
                hi = IMG_SIZE - 1 - MARGIN - half
                cx = random.randint(lo, hi)
                cy = random.randint(lo, hi)
                bbox = (cx - half, cy - half, cx + half, cy + half)
                if any(_bbox_overlaps(bbox, b, GAP) for b in placed_bboxes):
                    continue

                shape = random.choice(["circle", "rectangle"])
                colour = random.choice(colour_names)
                bbox = _draw_particle(img, shape, cx, cy, half, COLOURS_BGR[colour])
                placed_bboxes.append(bbox)
                area_px = _measured_area_px(img, bbox)
                particles.append({
                    "colour": colour,
                    "shape": shape,
                    "bbox": [int(v) for v in bbox],
                    "area_px": area_px,
                })
                break
            # If no placement found after 500 tries, just skip this particle.

        fname = f"img_{i}.png"
        cv2.imwrite(str(IMG_DIR / fname), img)
        ground_truth[fname] = {
            "count": len(particles),
            "particles": particles,
        }

    with GT_PATH.open("w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    return ground_truth


def main() -> None:
    gt = generate()
    total = sum(v["count"] for v in gt.values())
    print(f"Wrote {len(gt)} images to {IMG_DIR}")
    print(f"Wrote ground truth to {GT_PATH}")
    print(f"Total particles: {total}")
    for fname, entry in gt.items():
        print(f"  {fname}: count={entry['count']}")


if __name__ == "__main__":
    main()

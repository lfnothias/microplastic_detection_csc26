"""Find near-duplicate CorSeaCare photos.

Builds a labelled contact sheet of all photos and computes a perceptual hash (dHash) on
the sieve-interior crop (so the shared tray/mesh layout doesn't dominate). Prints
near-duplicate pairs by Hamming distance. Does NOT delete anything.

    .venv/bin/python scripts/find_redundant.py
"""
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "corseacare"
SHEET = ROOT / "data" / "corseacare_contact_sheet.png"
THUMB = 240
COLS = 5
HAMMING_DUP = 12          # <= this many differing bits => likely duplicate


def center_crop(img, frac=0.6):
    H, W = img.shape[:2]
    ch, cw = int(H * frac), int(W * frac)
    y, x = (H - ch) // 2, (W - cw) // 2
    return img[y:y + ch, x:x + cw]


def dhash(img, n=8):
    g = cv2.cvtColor(center_crop(img), cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (n + 1, n))
    return (g[:, 1:] > g[:, :-1]).flatten()


def main():
    paths = sorted(SRC.glob("*.JPG")) + sorted(SRC.glob("*.jpg"))
    hashes = []
    cells = []
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        hashes.append(dhash(img))
        H, W = img.shape[:2]
        s = THUMB / max(H, W)
        th = cv2.resize(img, (int(W * s), int(H * s)))
        cell = np.full((THUMB, THUMB, 3), 20, np.uint8)
        cell[:th.shape[0], :th.shape[1]] = th
        cv2.putText(cell, f"{i}: {p.name[:8]}", (4, THUMB - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cells.append(cell)

    rows = (len(cells) + COLS - 1) // COLS
    sheet = np.full((rows * THUMB, COLS * THUMB, 3), 20, np.uint8)
    for i, c in enumerate(cells):
        r, cc = divmod(i, COLS)
        sheet[r * THUMB:(r + 1) * THUMB, cc * THUMB:(cc + 1) * THUMB] = c
    cv2.imwrite(str(SHEET), sheet)

    print(f"{len(paths)} photos. Contact sheet -> {SHEET}\n")
    print("Near-duplicate pairs (Hamming distance on sieve-interior dHash):")
    found = False
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            d = int(np.count_nonzero(hashes[i] != hashes[j]))
            if d <= HAMMING_DUP:
                found = True
                print(f"  [{i}] {paths[i].name}  <->  [{j}] {paths[j].name}   (dist {d})")
    if not found:
        print("  (no pairs under threshold — all fairly distinct)")
    print("\nIndex -> file:")
    for i, p in enumerate(paths):
        print(f"  {i}: {p.name}")


if __name__ == "__main__":
    main()

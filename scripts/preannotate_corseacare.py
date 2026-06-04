"""Saturation-based pre-annotation for the real CorSeaCare Manta-net photos.

The plastics are saturated colours; the sieve mesh and metal tray are grey (low
saturation). Thresholding on HSV saturation isolates coloured particles cheaply
(no torch). Produces, for each photo in data/corseacare/:
  - a preview overlay (boxes) under data/corseacare_preann/overlays/
  - a Label Studio import file (predictions) data/corseacare_preann/ls_tasks.json
  - a draft YOLO dataset under data/corseacare_preann/yolo/ (single class 'fragment')

Candidate boxes are STARTING POINTS to correct in Label Studio (classify morphotype,
add missed grey/transparent particles, drop organic). Default predicted label: fragment.
"""
import json
import shutil
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "corseacare"
OUT = ROOT / "data" / "corseacare_preann"
PROC_MAX = 1600          # process at this max dimension for speed
SAT_THRESH = 80          # HSV saturation cutoff for "coloured"
VAL_MIN = 40             # ignore near-black
MIN_AREA_FRAC = 3e-5     # of processed-image area
MAX_AREA_FRAC = 2e-2     # drop huge blobs (rim reflections, big debris)


def _sieve_mask(proc):
    """Detect the circular sieve and return a filled-circle ROI mask.

    Hough runs on a small 500px thumbnail (fast); falls back to a generous central
    circle when detection fails (the sieve is roughly centred in these photos).
    """
    ph, pw = proc.shape[:2]
    roi = np.zeros((ph, pw), np.uint8)
    ds = 500.0 / max(ph, pw)
    small = cv2.resize(proc, (max(1, int(pw * ds)), max(1, int(ph * ds))))
    sh, sw = small.shape[:2]
    gray = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=sh,
                               param1=100, param2=40,
                               minRadius=int(0.22 * min(sh, sw)),
                               maxRadius=int(0.50 * min(sh, sw)))
    if circles is not None:
        cx, cy, r = (np.around(circles[0][0]).astype(float) / ds)
        cv2.circle(roi, (int(cx), int(cy)), int(r * 0.95), 255, -1)
    else:
        cv2.circle(roi, (pw // 2, ph // 2), int(0.42 * min(ph, pw)), 255, -1)
    return roi


def candidate_boxes(img_bgr):
    """Return list of (x1,y1,x2,y2) in full-resolution pixels."""
    H, W = img_bgr.shape[:2]
    scale = PROC_MAX / max(H, W) if max(H, W) > PROC_MAX else 1.0
    proc = cv2.resize(img_bgr, (int(W * scale), int(H * scale))) if scale != 1.0 else img_bgr
    ph, pw = proc.shape[:2]
    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((s > SAT_THRESH) & (v > VAL_MIN)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, _sieve_mask(proc))      # keep only inside the sieve
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = ph * pw
    boxes = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < MIN_AREA_FRAC * area or a > MAX_AREA_FRAC * area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        inv = 1.0 / scale
        boxes.append((int(x * inv), int(y * inv), int((x + w) * inv), int((y + h) * inv)))
    return boxes


def main():
    (OUT / "overlays").mkdir(parents=True, exist_ok=True)
    (OUT / "yolo" / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "yolo" / "labels").mkdir(parents=True, exist_ok=True)
    images = sorted(SRC.glob("*.JPG")) + sorted(SRC.glob("*.jpg"))
    tasks = []
    summary = []
    for p in images:
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        boxes = candidate_boxes(img)
        summary.append((p.name, len(boxes)))

        # preview overlay (downscaled for quick viewing)
        ov = img.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 255, 0), 3)
        s = 1000 / max(H, W)
        cv2.imwrite(str(OUT / "overlays" / p.name), cv2.resize(ov, (int(W * s), int(H * s))))

        # YOLO draft labels (single class 0 = fragment)
        shutil.copy(p, OUT / "yolo" / "images" / p.name)
        lines = [f"0 {((x1+x2)/2)/W:.6f} {((y1+y2)/2)/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}"
                 for (x1, y1, x2, y2) in boxes]
        (OUT / "yolo" / "labels" / f"{p.stem}.txt").write_text("\n".join(lines))

        # Label Studio prediction (image served via local files; see guide)
        results = []
        for (x1, y1, x2, y2) in boxes:
            results.append({
                "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                "original_width": W, "original_height": H,
                "value": {"x": 100*x1/W, "y": 100*y1/H,
                          "width": 100*(x2-x1)/W, "height": 100*(y2-y1)/H,
                          "rectanglelabels": ["fragment"]},
            })
        tasks.append({
            "data": {"image": f"/data/local-files/?d=corseacare/{p.name}"},
            "predictions": [{"model_version": "sat-preann-v1", "result": results}],
        })

    (OUT / "ls_tasks.json").write_text(json.dumps(tasks, indent=2))
    (OUT / "yolo" / "data.yaml").write_text(
        f"path: {OUT/'yolo'}\ntrain: images\nval: images\nnames: [fragment]\n")
    print("candidate boxes per image:")
    for name, n in summary:
        print(f"  {name}: {n}")
    print(f"total candidates: {sum(n for _, n in summary)}")
    print(f"overlays: {OUT/'overlays'} | LS tasks: {OUT/'ls_tasks.json'}")


if __name__ == "__main__":
    main()

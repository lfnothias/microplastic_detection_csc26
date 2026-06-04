"""SAM2 automatic-mask pre-annotation for the real CorSeaCare photos.

Uses SAM2's automatic mask generator to propose candidate particles — including grey /
white / transparent fragments the colour (saturation) method misses — restricted to the
sieve interior. Outputs overlays + Label Studio predictions + a draft YOLO label set under
data/corseacare_preann_sam2/.

Requires the SAM2.1 tiny checkpoint at data/models/sam2.1_hiera_tiny.pt.
Run with MPS fallback enabled:
    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/preannotate_sam2.py
"""
import json
import os
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "corseacare"
OUT = ROOT / "data" / "corseacare_preann_sam2"
CKPT = ROOT / "data" / "models" / "sam2.1_hiera_tiny.pt"
CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
PROC_MAX = 1024
MIN_AREA_FRAC = 3e-5
MAX_AREA_FRAC = 2e-2
LS_IMG_BASE = os.environ.get("CORSEACARE_LS_IMG_BASE", "http://localhost:8081/corseacare/")
DEVICE = os.environ.get("CORSEACARE_DEVICE", "mps")


def sieve_mask(proc):
    """Filled-circle ROI for the sieve (Hough on a 500px thumbnail; central fallback)."""
    ph, pw = proc.shape[:2]
    roi = np.zeros((ph, pw), np.uint8)
    ds = 500.0 / max(ph, pw)
    small = cv2.resize(proc, (max(1, int(pw * ds)), max(1, int(ph * ds))))
    sh, sw = small.shape[:2]
    gray = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=sh, param1=100,
                               param2=40, minRadius=int(0.22 * min(sh, sw)),
                               maxRadius=int(0.50 * min(sh, sw)))
    if circles is not None:
        cx, cy, r = (np.around(circles[0][0]).astype(float) / ds)
        cv2.circle(roi, (int(cx), int(cy)), int(r * 0.95), 255, -1)
    else:
        cv2.circle(roi, (pw // 2, ph // 2), int(0.42 * min(ph, pw)), 255, -1)
    return roi


def main():
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    model = build_sam2(CFG, str(CKPT), device=DEVICE)
    points = int(os.environ.get("CORSEACARE_SAM2_POINTS", "32"))
    mg = SAM2AutomaticMaskGenerator(model, points_per_side=points, pred_iou_thresh=0.7,
                                    stability_score_thresh=0.9, min_mask_region_area=40)

    (OUT / "overlays").mkdir(parents=True, exist_ok=True)
    (OUT / "yolo" / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "yolo" / "labels").mkdir(parents=True, exist_ok=True)
    images = sorted(SRC.glob("*.JPG")) + sorted(SRC.glob("*.jpg"))
    _lim = int(os.environ.get("CORSEACARE_SAM2_LIMIT", "0"))
    if _lim:
        images = images[:_lim]
    tasks, summary = [], []
    for p in images:
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        scale = PROC_MAX / max(H, W) if max(H, W) > PROC_MAX else 1.0
        proc = cv2.resize(img, (int(W * scale), int(H * scale))) if scale != 1.0 else img
        ph, pw = proc.shape[:2]
        roi = sieve_mask(proc)
        masks = mg.generate(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
        area = ph * pw
        boxes = []
        for m in masks:
            a = m["area"]
            if a < MIN_AREA_FRAC * area or a > MAX_AREA_FRAC * area:
                continue
            x, y, w, h = m["bbox"]
            cy, cx = int(min(ph - 1, y + h / 2)), int(min(pw - 1, x + w / 2))
            if roi[cy, cx] == 0:
                continue
            inv = 1.0 / scale
            boxes.append((int(x * inv), int(y * inv), int((x + w) * inv), int((y + h) * inv)))
        summary.append((p.name, len(boxes)))

        ov = img.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 255, 0), 3)
        s = 1000 / max(H, W)
        cv2.imwrite(str(OUT / "overlays" / p.name), cv2.resize(ov, (int(W * s), int(H * s))))

        import shutil
        shutil.copy(p, OUT / "yolo" / "images" / p.name)
        lines = [f"0 {((x1+x2)/2)/W:.6f} {((y1+y2)/2)/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}"
                 for (x1, y1, x2, y2) in boxes]
        (OUT / "yolo" / "labels" / f"{p.stem}.txt").write_text("\n".join(lines))

        results = [{
            "from_name": "label", "to_name": "image", "type": "rectanglelabels",
            "original_width": W, "original_height": H,
            "value": {"x": 100*x1/W, "y": 100*y1/H, "width": 100*(x2-x1)/W,
                      "height": 100*(y2-y1)/H, "rectanglelabels": ["fragment"]},
        } for (x1, y1, x2, y2) in boxes]
        tasks.append({"data": {"image": f"{LS_IMG_BASE}{p.name}"},
                      "predictions": [{"model_version": "sam2-auto-v1", "result": results}]})

    (OUT / "ls_tasks.json").write_text(json.dumps(tasks, indent=2))
    (OUT / "yolo" / "data.yaml").write_text(
        f"path: {OUT/'yolo'}\ntrain: images\nval: images\nnames: [fragment]\n")
    for name, n in summary:
        print(f"  {name}: {n}")
    print(f"total SAM2 candidates: {sum(n for _, n in summary)}")
    print(f"overlays: {OUT/'overlays'}")


if __name__ == "__main__":
    main()

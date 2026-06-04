"""Fine-tune YOLO11n locally on the Apple M4 (MPS) on the synthetic test dataset.

This validates the *real* training path on-device. Run gt_to_yolo.py first.
Outputs (weights, plots) go under data/runs/ (gitignored).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train_synth_yolo" / "data.yaml"
RUNS = ROOT / "data" / "runs"


def main():
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    model.train(data=str(DATA), epochs=40, imgsz=416, batch=16, device="mps",
                project=str(RUNS), name="local_poc", exist_ok=True, verbose=False)
    metrics = model.val(data=str(DATA), device="mps", verbose=False)
    print("VAL mAP50:", float(metrics.box.map50), "| mAP50-95:", float(metrics.box.map))
    print("BEST:", RUNS / "local_poc" / "weights" / "best.pt")


if __name__ == "__main__":
    main()

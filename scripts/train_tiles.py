"""Fine-tune YOLO11n on the TILED real dataset (Apple M4 / MPS).

Run `export_from_ls.py` then `tile_dataset.py` first (they build data/ls_tiles_yolo/).
Early-stops with patience to avoid the small-data overfitting tail. Outputs (weights, plots)
go under data/runs/<name>/ (gitignored).

    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/train_tiles.py [name] [epochs] [patience]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ls_tiles_yolo" / "data.yaml"
RUNS = ROOT / "data" / "runs"


def main(name="tiles_v3", epochs=120, patience=25):
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    model.train(data=str(DATA), epochs=epochs, imgsz=640, batch=16, device="mps",
                patience=patience, project=str(RUNS), name=name, exist_ok=True, verbose=False)
    m = model.val(data=str(DATA), device="mps", verbose=False)
    print(f"VAL mAP50: {float(m.box.map50):.3f} mAP50-95: {float(m.box.map):.3f}")
    maps = list(m.box.maps)
    for i, nm in sorted(model.names.items()):
        if i < len(maps):
            print(f"  {nm}: mAP50-95={maps[i]:.3f}")
    print("BEST:", RUNS / name / "weights" / "best.pt")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "tiles_v3"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    patience = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    main(name, epochs, patience)

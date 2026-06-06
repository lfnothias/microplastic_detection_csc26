"""Quick standalone YOLO val on the current tiled dataset. Internal helper for results reporting.
    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/_val.py <weights.pt>
"""
import sys
from ultralytics import YOLO

m = YOLO(sys.argv[1])
r = m.val(data="data/ls_tiles_yolo/data.yaml", device="mps", verbose=False, plots=False)
maps = list(r.box.maps)
pc = {m.names[i]: round(maps[i], 3) for i in sorted(m.names) if i < len(maps)}
print(f"RESULT mAP50={float(r.box.map50):.3f} mAP50-95={float(r.box.map):.3f} "
      f"P={float(r.box.mp):.3f} R={float(r.box.mr):.3f} perclass={pc}")

from dataclasses import dataclass, field
from pathlib import Path
import yaml

DEFAULT_CLASSES = ["fragment", "fibre", "film", "mousse", "pellet", "autre"]


@dataclass
class Config:
    classes: list[str] = field(default_factory=lambda: list(DEFAULT_CLASSES))
    mm_per_px: float = 0.1            # scale factor; MUST be calibrated per camera setup
    weights: str = "yolo11n.pt"       # detector checkpoint
    sam2_checkpoint: str = "sam2_hiera_tiny.pt"
    sam2_model_cfg: str = "sam2_hiera_t.yaml"
    conf_threshold: float = 0.25


def load_config(path: str | Path) -> Config:
    data = yaml.safe_load(Path(path).read_text()) or {}
    cfg = Config()
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg

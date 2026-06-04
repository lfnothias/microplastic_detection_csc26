from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]   # x1, y1, x2, y2 in pixels
    class_id: int
    class_name: str
    confidence: float


@dataclass(frozen=True)
class SampleMetadata:
    date: str = ""
    gps_start: str = ""
    gps_end: str = ""
    location: str = ""
    weather: str = ""
    sea_state: str = ""
    boat_speed_kn: Optional[float] = None
    operator: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParticleRecord:
    class_name: str
    confidence: float
    colour: str
    area_mm2: float
    max_feret_mm: float
    area_px: int
    xyxy: tuple[float, float, float, float]
    extra: dict = field(default_factory=dict)

    def to_row(self, sample: Optional[SampleMetadata] = None) -> dict:
        row = {
            "class_name": self.class_name, "confidence": self.confidence,
            "colour": self.colour, "area_mm2": self.area_mm2,
            "max_feret_mm": self.max_feret_mm, "area_px": self.area_px,
            "x1": self.xyxy[0], "y1": self.xyxy[1], "x2": self.xyxy[2], "y2": self.xyxy[3],
        }
        if sample is not None:
            row.update(sample.to_dict())
        return row

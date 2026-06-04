import numpy as np
from corseacare.types import Detection, SampleMetadata
from corseacare.measure import measure_particle, assemble_records


def _red_img_mask():
    img = np.zeros((20, 20, 3), np.uint8); img[:, :, 2] = 255
    mask = np.zeros((20, 20), np.uint8); mask[5:15, 5:15] = 1
    return img, mask


def test_measure_particle_record():
    img, mask = _red_img_mask()
    det = Detection(xyxy=(5, 5, 15, 15), class_id=0, class_name="fragment", confidence=0.8)
    rec = measure_particle(img, mask, det, mm_per_px=0.5)
    assert rec.class_name == "fragment"
    assert rec.colour == "rouge"
    assert rec.area_mm2 == 25.0


def test_assemble_records_attaches_metadata():
    img, mask = _red_img_mask()
    det = Detection(xyxy=(5, 5, 15, 15), class_id=0, class_name="fragment", confidence=0.8)
    meta = SampleMetadata(location="Lisula", boat_speed_kn=2.5)
    rows = assemble_records(img, [det], [mask], mm_per_px=0.5, sample=meta)
    assert len(rows) == 1
    assert rows[0]["location"] == "Lisula"
    assert rows[0]["colour"] == "rouge"

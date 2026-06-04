from corseacare.types import Detection, SampleMetadata


def test_detection_fields():
    d = Detection(xyxy=(1.0, 2.0, 3.0, 4.0), class_id=0, class_name="fragment", confidence=0.9)
    assert d.class_name == "fragment"
    assert d.xyxy[2] == 3.0


def test_sample_metadata_to_dict():
    m = SampleMetadata(date="2026-06-04", gps_start="42.0,8.7", gps_end="42.1,8.8",
                       location="Lisula", weather="clear", sea_state="calm",
                       boat_speed_kn=2.5, operator="LFX")
    d = m.to_dict()
    assert d["location"] == "Lisula" and d["boat_speed_kn"] == 2.5

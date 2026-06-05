from corseacare.ls_merge import apply_class_suggestions


def test_apply_class_suggestions():
    tasks = [{"data": {"image": "http://x/a.JPG"},
              "predictions": [{"model_version": "sat", "result": [
                  {"from_name": "label", "to_name": "image", "type": "rectanglelabels",
                   "value": {"x": 1, "y": 1, "width": 5, "height": 5, "rectanglelabels": ["fragment"]}}]}]}]
    out = apply_class_suggestions(tasks, {"a.JPG": {0: "fibre"}})
    assert out[0]["predictions"][0]["result"][0]["value"]["rectanglelabels"] == ["fibre"]
    # original not mutated
    assert tasks[0]["predictions"][0]["result"][0]["value"]["rectanglelabels"] == ["fragment"]

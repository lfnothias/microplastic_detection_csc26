"""Apply per-box class suggestions (e.g. from Claude few-shot) into Label Studio tasks."""
import copy


def apply_class_suggestions(tasks, suggestions):
    """suggestions: {image_filename: {box_index: class_name}}. Returns an updated copy."""
    out = copy.deepcopy(tasks)
    for task in out:
        name = task["data"]["image"].rsplit("/", 1)[-1]
        per_box = suggestions.get(name, {})
        for i, r in enumerate(task["predictions"][0]["result"]):
            if i in per_box:
                r["value"]["rectanglelabels"] = [per_box[i]]
    return out

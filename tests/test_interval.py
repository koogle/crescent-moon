from crescent_moon.metrics import interval


def record(image, predicted, boxes, kind="present"):
    return {
        "dataset": "d",
        "image_id": image,
        "kind": kind,
        "exhaustive": True,
        "predicted": predicted,
        "boxes": boxes,
    }


def test_interval_brackets_the_score_and_counts_absent_boxes_as_wrong():
    box, other = [0, 0, 10, 10], [50, 50, 60, 60]
    records = [record(i, [box], [box]) for i in range(40)]
    records += [record(100 + i, [other], [box]) for i in range(10)]
    records += [record(200, [box], [], kind="absent")]
    bounds = interval(records, repeats=500)
    assert bounds["recall"][0] < 0.8 < bounds["recall"][1]
    assert bounds["precision"][1] <= 1.0
    assert bounds["f1"][0] < bounds["f1"][1]

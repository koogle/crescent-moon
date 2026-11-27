from crescent_moon.tasks import coco_samples


def rows(count):
    return [
        {
            "image_id": i,
            "width": 100,
            "height": 100,
            "objects": {"category": [0], "bbox": [[0, 0, 10, 10]]},
        }
        for i in range(count)
    ]


def test_exclude_samples_leaves_the_tuning_split_out():
    data = rows(60)
    test = {s["image_id"] for s in coco_samples(data, 10, 7)}
    tune = {s["image_id"] for s in coco_samples(data, 10, 7, exclude_images=10)}
    full = {
        s["image_id"]
        for s in coco_samples(data, 1000, 7, exclude_samples=[{"images": 10, "exclude_images": 10}])
    }
    assert not tune & full
    assert test <= full
    assert len(full) == 50

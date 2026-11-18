from crescent_moon.run import predict
from crescent_moon.tasks import (
    COCO_NAMES,
    coco_json_samples,
    coco_samples,
    labelled_samples,
    lvis_samples,
    refcoco_m_samples,
)

LVIS = {
    "categories": [
        {"id": 1, "name": "bow_(weapon)", "frequency": "r"},
        {"id": 2, "name": "cup", "frequency": "f"},
        {"id": 3, "name": "zebra", "frequency": "c"},
        {"id": 4, "name": "kite", "frequency": "c"},
    ],
    "images": [
        {
            "id": 7,
            "coco_url": "http://images.cocodataset.org/val2017/7.jpg",
            "width": 640,
            "height": 480,
            "neg_category_ids": [3, 4],
            "not_exhaustive_category_ids": [2],
        },
        {
            "id": 8,
            "coco_url": "http://images.cocodataset.org/val2017/8.jpg",
            "width": 640,
            "height": 480,
            "neg_category_ids": [],
            "not_exhaustive_category_ids": [],
        },
    ],
    "annotations": [
        {"image_id": 7, "category_id": 1, "bbox": [10, 20, 30, 40]},
        {"image_id": 7, "category_id": 2, "bbox": [0, 0, 5, 5]},
        {"image_id": 7, "category_id": 2, "bbox": [50, 50, 5, 5]},
    ],
}


def test_lvis_prompts_present_and_verified_absent_categories():
    (sample,) = lvis_samples(LVIS, images=5, max_absent=1, seed=0)  # image 8 has no boxes
    assert sample["image_id"] == 7
    bow, cup, absent = sample["queries"]
    assert (bow["phrase"], bow["boxes"], bow["exhaustive"]) == (
        "bow (weapon)",
        [[10, 20, 40, 60]],
        True,
    )
    assert (len(cup["boxes"]), cup["exhaustive"], cup["group"]) == (2, False, "f")
    assert absent["kind"] == "absent" and absent["phrase"] in {"zebra", "kite"}
    assert len(lvis_samples(LVIS, 5, 0, seed=0)[0]["queries"]) == 2
    assert lvis_samples(LVIS, 5, 1, seed=0) == lvis_samples(LVIS, 5, 1, seed=0)


def test_coco_prompts_the_categories_in_the_image():
    rows = [
        {
            "image_id": 139,
            "width": 640,
            "height": 426,
            "objects": {"category": [1, 0, 1], "bbox": [[0, 0, 5, 5], [1, 1, 9, 9], [2, 2, 4, 4]]},
        },
        {"image_id": 140, "width": 1, "height": 1, "objects": {"category": [], "bbox": []}},
    ]
    (sample,) = coco_samples(rows, images=5, seed=0)
    assert len(COCO_NAMES) == 80
    assert sample["image"] == "coco/139.jpg"
    person, bicycle = sample["queries"]
    assert (person["phrase"], person["boxes"]) == ("person", [[1, 1, 9, 9]])
    assert (bicycle["phrase"], len(bicycle["boxes"])) == ("bicycle", 2)


def test_refcoco_m_asks_every_expression_of_every_referred_object():
    rows = [
        {
            "image_id": 99,
            "samples": [
                {
                    "sentences": ["left guy ", "person on left"],
                    "bbox": [10, 20, 30, 40],
                    "mask": {"size": [294, 500]},
                },
                {
                    "sentences": ["right guy"],
                    "bbox": [300, 0, 100, 200],
                    "mask": {"size": [294, 500]},
                },
            ],
        }
    ]
    (sample,) = refcoco_m_samples(rows, images=10, seed=0)
    assert (sample["width"], sample["height"], sample["image"]) == (500, 294, "refcoco-m/99.jpg")
    assert [q["phrase"] for q in sample["queries"]] == ["left guy", "person on left", "right guy"]
    assert sample["queries"][0]["boxes"] == [[10, 20, 40, 60]]
    assert all(q["kind"] == "referring" for q in sample["queries"])


def test_predict_joins_detections_to_queries(tmp_path):
    from PIL import Image

    (sample,) = lvis_samples(LVIS, images=5, max_absent=0, seed=0)
    sample["dataset"] = "lvis_minival"
    cached = tmp_path / "images.cocodataset.org/val2017/7.jpg"
    cached.parent.mkdir(parents=True)
    Image.new("RGB", (64, 48)).save(cached)

    class Detector:
        calls = 0

        def detect(self, image, phrases):
            self.calls += 1
            return {
                "boxes": [[[0, 0, 1, 1]] for _ in phrases],
                "shared_seconds": 0.5,
                "phrase_seconds": [0.1] * len(phrases),
            }

    detector = Detector()
    (records,) = predict(detector, [sample], tmp_path, warmup=1)
    assert detector.calls == 2  # the warm-up pass is not recorded
    assert [r["phrase"] for r in records] == ["bow (weapon)", "cup"]
    assert all(r["predicted"] == [[0, 0, 1, 1]] and r["dataset"] == "lvis_minival" for r in records)


def test_above_keeps_boxes_at_or_over_the_threshold():
    from crescent_moon.run import above

    record = {"predicted": [[0, 0, 1, 1], [1, 1, 2, 2]], "scores": [0.4, 0.9]}
    assert above([record], 0.5)[0]["predicted"] == [[1, 1, 2, 2]]
    assert above([record], 0.4)[0]["predicted"] == record["predicted"]


def test_coco_tuning_split_does_not_overlap_the_evaluation_split():
    rows = [
        {
            "image_id": i,
            "width": 10,
            "height": 10,
            "objects": {"category": [0], "bbox": [[0, 0, 5, 5]]},
        }
        for i in range(40)
    ]
    evaluation = {s["image_id"] for s in coco_samples(rows, images=10, seed=3)}
    tuning = {s["image_id"] for s in coco_samples(rows, images=10, seed=3, exclude_images=10)}
    assert len(evaluation) == len(tuning) == 10 and not evaluation & tuning
    # The evaluation split itself is unchanged by the option's existence.
    assert evaluation == {s["image_id"] for s in coco_samples(rows, 10, 3)}


def test_voc_annotation_and_samples():
    from crescent_moon.tasks import voc_annotation, voc_samples

    xml = b"""<annotation><filename>000001.jpg</filename>
    <size><width>100</width><height>80</height><depth>3</depth></size>
    <object><name>tvmonitor</name><difficult>0</difficult>
      <bndbox><xmin>1</xmin><ymin>11</ymin><xmax>50</xmax><ymax>60</ymax></bndbox></object>
    <object><name>person</name><difficult>1</difficult>
      <bndbox><xmin>5</xmin><ymin>5</ymin><xmax>9</xmax><ymax>9</ymax></bndbox></object>
    </annotation>"""
    row = voc_annotation(xml)
    assert row["image_id"] == "000001" and (row["width"], row["height"]) == (100, 80)
    assert row["objects"][0] == {"name": "tv", "box": [0.0, 10.0, 50.0, 60.0], "difficult": False}
    (sample,) = voc_samples([row], images=5, seed=0)
    assert sample["image"] == "voc2007/000001.jpg"
    assert [(q["phrase"], q["group"]) for q in sample["queries"]] == [
        ("person", "difficult"),
        ("tv", ""),
    ]


def test_mot_samples_take_people_on_every_nth_frame():
    from crescent_moon.tasks import mot_samples

    lines = [
        "1,1,11,21,10,20,1,1,1.0",  # pedestrian
        "1,2,51,51,5,5,0,7,1.0",  # static person: a person too
        "1,3,61,61,5,5,0,12,1.0",  # reflection: not a person
        "2,1,12,21,10,20,1,1,1.0",
        "4,1,14,21,10,20,1,1,1.0",
    ]
    truth = "\n".join(lines)
    samples = mot_samples("MOT17-09", truth, 1920, 1080, every=3)
    assert [s["image_id"] for s in samples] == ["MOT17-09/000001", "MOT17-09/000004"]
    assert samples[0]["queries"][0]["boxes"] == [[10.0, 20.0, 20.0, 40.0], [50.0, 50.0, 55.0, 55.0]]
    assert samples[0]["image"] == "mot17/MOT17-09/000001.jpg"


def test_coco_samples_ask_a_set_number_of_categories():
    rows = [
        {
            "image_id": 1,
            "width": 10,
            "height": 10,
            "objects": {"category": [3, 5], "bbox": [[0, 0, 1, 1], [1, 1, 2, 2]]},
        }
    ]
    (sample,) = coco_samples(rows, images=1, seed=0, categories=6)
    kinds = [q["kind"] for q in sample["queries"]]
    assert kinds == ["present", "present", "absent", "absent", "absent", "absent"]
    assert len({q["phrase"] for q in sample["queries"]}) == 6
    (one,) = coco_samples(rows, images=1, seed=0, categories=1)
    assert [q["phrase"] for q in one["queries"]] == [COCO_NAMES[3]]
    assert coco_samples(rows, images=1, seed=0) == coco_samples(rows, images=1, seed=0)


def test_coco_json_samples_group_boxes_by_category():
    annotations = {
        "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
        "images": [
            {"id": 7, "file_name": "a.jpeg", "width": 100, "height": 50},
            {"id": 8, "file_name": "b.jpeg", "width": 100, "height": 50},
        ],
        "annotations": [
            {"image_id": 7, "category_id": 3, "bbox": [10, 5, 20, 10]},
            {"image_id": 7, "category_id": 1, "bbox": [0, 0, 5, 5]},
            {"image_id": 7, "category_id": 3, "bbox": [50, 5, 10, 10]},
        ],
    }
    (sample,) = coco_json_samples(annotations, images=5, seed=0, folder="set")
    assert sample["image"] == "set/a.jpeg"
    assert [q["phrase"] for q in sample["queries"]] == ["car", "person"]
    assert sample["queries"][0]["boxes"] == [[10, 5, 30, 15], [50, 5, 60, 15]]


def test_labelled_samples_prompt_the_labelled_categories():
    replies = [
        {"image_id": "generated/00001", "error": "timeout"},
        {
            "image_id": "generated/00002",
            "width": 10,
            "height": 8,
            "objects": [
                {"label": "dog", "box": [0, 0, 4, 4]},
                {"label": "dog", "box": [5, 5, 9, 7]},
            ],
        },
    ]
    (sample,) = labelled_samples(replies, images=5, seed=0, folder="/data")
    assert sample["image"] == "/data/generated/00002.jpg"
    assert sample["queries"][0]["phrase"] == "dog" and len(sample["queries"][0]["boxes"]) == 2


def test_generated_scene_fills_the_template_and_agrees_in_number():
    import random

    from crescent_moon.images import plural, scene

    assert [plural(n) for n in ("person", "bus", "knife", "skis", "couch", "cup")] == [
        "people",
        "buses",
        "knives",
        "skis",
        "couches",
        "cups",
    ]
    settings = {
        "max_categories": 2,
        "counts": ["two"],
        "groups": [["oven", "sink"]],
        "group_share": 1.0,
        "attributes": ["old"],
        "attribute_share": 1.0,
        "template": "{medium} of {things} {place}",
        "choices": {"medium": ["A sketch"], "place": ["", ""]},
    }
    prompt, asked = scene(settings, random.Random(0))
    assert set(asked) <= {"oven", "sink"}
    assert prompt.startswith("A sketch of two old ") and not prompt.endswith(" ")

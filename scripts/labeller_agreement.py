"""Rescore stored predictions on teacher-labelled images against other labellers.

    python scripts/labeller_agreement.py --spec artifacts/agreement.json > artifacts/agreement-scores.json

The spec is {"dataset", "labels": {labeller: JSONL of crescent_moon.teacher},
"runs": [{"name", "predictions", "threshold"?}]}. The predictions were recorded for the
categories the first labeller named in each image, so every truth here is restricted to
those: a labeller's truth is its boxes for the categories asked, and "agreed" is the
boxes at least two labellers gave (`teacher.consensus`). Only images every labeller
answered are scored. Output: one entry per run and truth with precision, recall and F1 at
IoU 0.5 and a bootstrap interval of the F1.
"""

import argparse
import json
from pathlib import Path

from crescent_moon.metrics import interval, summarize
from crescent_moon.run import above, read_records
from crescent_moon.teacher import consensus


def replies(path):
    """{image_id: objects} of one labeller's answered images."""
    lines = (json.loads(line) for line in Path(path).read_text().splitlines())
    return {r["image_id"]: r["objects"] for r in lines if "error" not in r}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    labels = {name: replies(path) for name, path in spec["labels"].items()}
    common = set.intersection(*(set(v) for v in labels.values()))
    truths = {name: {i: objects[i] for i in common} for name, objects in labels.items()}
    truths["agreed"] = {i: consensus({n: labels[n][i] for n in labels}, 2) for i in common}
    out = []
    for run in spec["runs"]:
        records = [
            r
            for r in read_records(Path(run["predictions"]))
            if r["dataset"] == spec["dataset"] and r["image_id"] in common
        ]
        if run.get("threshold") is not None:
            records = above(records, run["threshold"])
        for truth, objects in truths.items():
            rescored = [
                r
                | {"boxes": [o["box"] for o in objects[r["image_id"]] if o["label"] == r["phrase"]]}
                for r in records
            ]
            # A category this truth lacks in an image is asked of it all the same.
            rescored = [r | {"kind": "present" if r["boxes"] else "absent"} for r in rescored]
            scores = summarize(rescored)[spec["dataset"]]["detection"]["iou_0.5"]
            out.append(
                {
                    "name": run["name"],
                    "truth": truth,
                    "images": len(common),
                    "truth_boxes": sum(len(r["boxes"]) for r in rescored),
                    **{k: scores[k] for k in ("precision", "recall", "f1")},
                    "f1_interval_95": interval(rescored)["f1"],
                }
            )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

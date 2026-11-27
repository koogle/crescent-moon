"""Score stored predictions: F1 with a bootstrap interval, and average precision.

    python scripts/score_runs.py --runs artifacts/full-runs.json > artifacts/full-scores.json

`--runs` is a JSON list of {"name", "predictions", "dataset", and optionally "threshold",
"aliases", "ap"}. Each entry's records of that dataset are cut at `threshold` when given
(for runs recorded at a low confidence floor), and scored with `metrics.summarize` and
`metrics.interval`. With "ap" the entry also gets the mean over categories of the area
under the precision-recall curve, at IoU 0.5 and averaged over IoU 0.5 to 0.95, counting
boxes for categories absent from an image as misses; this is COCO's AP when every
category was asked of every image. `aliases` maps a phrase to the labels that answer it
for records that keep every candidate label. With "config" (the evaluation config, for
image sizes) the entry also gets recall by object size in cells of the vision encoder's grid.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from crescent_moon.metrics import interval, ranked_ap, summarize
from crescent_moon.run import above, read_records

sys.path.insert(0, str(Path(__file__).parent))
from study_selection import image_sizes, recall_by_cells


def score(entry):
    records = [
        r for r in read_records(Path(entry["predictions"])) if r["dataset"] == entry["dataset"]
    ]
    out = {"name": entry["name"], "dataset": entry["dataset"], "threshold": entry.get("threshold")}
    if entry.get("ap"):
        out["ap_iou_0.5"] = ranked_ap(records, 0.5, absent=True)["category_mean_ap"]
        out["ap_iou_0.5_to_0.95"] = float(
            np.mean(
                [
                    ranked_ap(records, t, absent=True)["category_mean_ap"]
                    for t in np.arange(0.5, 0.96, 0.05)
                ]
            )
        )
    kept = above(records, entry["threshold"]) if entry.get("threshold") is not None else records
    present = [r for r in kept if r["kind"] != "absent"]
    summary = summarize(present)[entry["dataset"]]
    detection = summary["detection"]
    out |= {
        "images": len({r["image_id"] for r in present}),
        "iou_0.5": {k: detection["iou_0.5"][k] for k in ("precision", "recall", "f1")},
        "f1_iou_0.75": detection["iou_0.75"]["f1"],
        "interval_95": interval(present),
        "image_seconds": summary["latency"]["image_seconds"]["mean"],
    }
    if entry.get("config"):
        sizes = image_sizes([entry["config"]], Path("artifacts/images"))
        sized = [r | {"size": sizes[r["dataset"], r["image_id"]]} for r in present]
        out["recall_by_cells"] = recall_by_cells(sized)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps([score(entry) for entry in json.loads(args.runs.read_text())], indent=2))


if __name__ == "__main__":
    main()

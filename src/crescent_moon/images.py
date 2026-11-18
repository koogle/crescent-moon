"""Training images from Open Images, Objects365 and COCO, for teacher labelling.

    python -m crescent_moon.images --config configs/teacher-images.json \
        --out artifacts/teacher-images

Writes `<out>/<source>/<id>.jpg`, with the longest side at most `max_side`, and appends
one line per image to `<out>/manifest.jsonl` ({"image_id", "image", "width", "height",
"source"}, with "image" relative to `<out>`). A source that already has its quota in
the manifest is skipped, so an interrupted run continues.

- Open Images (`vikhyatk/openimages-bbox`, Parquet): `per_shard` seeded images from
  each of `shards` seeded training shards, one shard on disk at a time.
- Objects365 (`guozonghao96/objects365`, tar.gz patches): every `stride`-th image of
  each listed patch, streamed without saving the archive, until the quota is reached.
- COCO (`detection-datasets/coco`, Parquet): every image with at least one annotated
  object in the listed training shards, in shard order, named by its COCO image id, so
  a teacher's labels can be set against COCO's own on the same images.
- Generated (a source with a "model", a text-to-image model on fal.ai): one image per
  prompt, which names a few COCO categories and fills the config's template from its lists
  of media, places and so on (`scene`); the manifest line keeps the prompt and the
  categories it names.
"""

import argparse
import io
import json
import random
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image

HUB = "https://huggingface.co/datasets"


def save(data, out, source, name, max_side, existing):
    """Write a resized RGB JPEG; return its manifest line, or None if unreadable or
    already in the manifest."""
    if f"{source}/{name}" in existing:
        return None
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (OSError, ValueError):
        return None
    image.thumbnail((max_side, max_side))
    path = Path(source) / f"{name}.jpg"
    (out / path).parent.mkdir(parents=True, exist_ok=True)
    image.save(out / path, format="JPEG", quality=90)
    width, height = image.size
    existing.add(f"{source}/{name}")
    return {
        "image_id": f"{source}/{name}",
        "image": str(path),
        "width": width,
        "height": height,
        "source": source,
    }


def open_images(settings, out, quota, max_side, rng, existing):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    shards = rng.sample(range(settings["total_shards"]), settings["shards"])
    for shard in sorted(shards):
        if quota <= 0:
            return
        name = f"data/train-{shard:05d}-of-{settings['total_shards']:05d}.parquet"
        path = Path(
            hf_hub_download(
                settings["repo"],
                name,
                repo_type="dataset",
                revision=settings["revision"],
                cache_dir=out / ".download",
            )
        )
        table = pq.read_table(path, columns=["image"])
        rows = rng.sample(range(table.num_rows), min(settings["per_shard"], table.num_rows, quota))
        images = table.column("image")
        for row in sorted(rows):
            line = save(
                images[row].as_py()["bytes"],
                out,
                "openimages",
                f"{shard}-{row}",
                max_side,
                existing,
            )
            if line:
                quota -= 1
                yield line
        # The cache entry is a symlink to a blob; remove both to keep one shard on disk.
        path.resolve().unlink()
        path.unlink()


def objects365(settings, out, quota, max_side, existing):
    for patch in settings["patches"]:
        url = f"{HUB}/{settings['repo']}/resolve/{settings['revision']}/{patch}"
        # A dropped connection restarts the patch; images already saved are skipped.
        for _attempt in range(4):
            if quota <= 0:
                return
            try:
                for line in patch_images(url, settings["stride"], out, max_side, existing):
                    quota -= 1
                    yield line
                    if quota <= 0:
                        return
                break
            except (tarfile.ReadError, OSError, EOFError):
                continue


def patch_images(url, stride, out, max_side, existing):
    """Every `stride`-th image of one streamed tar.gz patch, saved and resized."""
    with urllib.request.urlopen(url, timeout=600) as response:
        seen = 0
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".jpg"):
                    continue
                seen += 1
                if seen % stride:
                    continue
                data = archive.extractfile(member).read()
                line = save(data, out, "objects365", Path(member.name).stem, max_side, existing)
                if line:
                    yield line


def coco(settings, out, quota, max_side, existing):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for name in settings["files"]:
        path = hf_hub_download(
            settings["repo"], name, repo_type="dataset", revision=settings["revision"]
        )
        table = pq.ParquetFile(path)
        for group in range(table.metadata.num_row_groups):
            rows = table.read_row_group(group, columns=["image_id", "image", "objects"])
            for row in rows.to_pylist():
                if quota <= 0:
                    return
                if not row["objects"]["bbox"]:
                    continue
                line = save(row["image"]["bytes"], out, "coco", row["image_id"], max_side, existing)
                if line is not None:
                    quota -= 1
                    yield line


FAL = "https://fal.run/"


PLURALS = {"person": "people", "knife": "knives", "mouse": "mice", "sheep": "sheep"}


def plural(name):
    """A COCO category name in the plural ("skis" and "scissors" already are)."""
    if name in PLURALS or name.endswith(("skis", "scissors")):
        return PLURALS.get(name, name)
    return name + ("es" if name.endswith(("s", "x", "ch", "sh")) else "s")


def scene(settings, rng):
    """One prompt of a generated source and the categories it names.

    The prompt is `settings["template"]` with `{things}` and one drawn phrase per list in
    `settings["choices"]` (a phrase list may hold "" for nothing). The things are one to
    `max_categories` COCO categories, each with a count from `counts` and, with
    probability `attribute_share`, a modifier from `attributes`. With probability
    `group_share` the categories come from one of `groups` (categories that occur
    together), otherwise from all 80."""
    from crescent_moon.tasks import COCO_NAMES

    many = rng.randint(1, settings["max_categories"])
    groups = settings.get("groups")
    if groups and rng.random() < settings["group_share"]:
        pool = rng.choice(groups)
        asked = rng.sample(pool, min(many, len(pool)))
    else:
        asked = rng.sample(COCO_NAMES, many)
    things = []
    for name in asked:
        count = rng.choice(settings["counts"])
        modifier = ""
        if settings.get("attributes") and rng.random() < settings["attribute_share"]:
            modifier = rng.choice(settings["attributes"]) + " "
        # "a" is the only count that takes a singular noun, and its article agrees.
        if count == "a":
            count = "an" if (modifier + name)[0] in "aeiou" else "a"
            things.append(f"{count} {modifier}{name}")
        else:
            things.append(f"{count} {modifier}{plural(name)}")
    listed = things[0] if len(things) == 1 else ", ".join(things[:-1]) + " and " + things[-1]
    drawn = {key: rng.choice(phrases) for key, phrases in settings["choices"].items()}
    prompt = " ".join(settings["template"].format(things=listed, **drawn).split())
    return prompt.replace(" ,", ","), asked


def generated(source, settings, out, quota, max_side, rng, existing):
    """Images from a text-to-image model on fal.ai (`settings["model"]`), one per `scene`,
    in a size drawn from `sizes`; the manifest line keeps the prompt and the categories
    it names. The key is read from the FAL_KEY environment variable."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    key = os.environ["FAL_KEY"]
    scenes = [
        (*scene(settings, rng), rng.choice(settings["sizes"])) for _ in range(settings["images"])
    ]
    wanted = [(i, s) for i, s in enumerate(scenes) if f"{source}/{i:05d}" not in existing]

    def one(job):
        index, (prompt, _, size) = job
        body = {
            "prompt": prompt,
            "seed": index,
            "num_images": 1,
            "image_size": size,
            **settings["request"],
        }
        request = urllib.request.Request(
            FAL + settings["model"],
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                url = json.load(response)["images"][0]["url"]
            with urllib.request.urlopen(url, timeout=180) as response:
                return response.read()
        except (OSError, KeyError, IndexError, ValueError):
            return None

    with ThreadPoolExecutor(settings.get("workers", 8)) as pool:
        for (index, (prompt, asked, _)), data in zip(
            wanted[:quota], pool.map(one, wanted[:quota]), strict=True
        ):
            line = data and save(data, out, source, f"{index:05d}", max_side, existing)
            if line:
                yield line | {"prompt": prompt, "asked": asked}


def prepare(config, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    counts, existing = {}, set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            record = json.loads(line)
            counts[record["source"]] = counts.get(record["source"], 0) + 1
            existing.add(record["image_id"])
    rng = random.Random(config["seed"])
    makers = {
        "openimages": lambda s, q: open_images(s, out, q, config["max_side"], rng, existing),
        "objects365": lambda s, q: objects365(s, out, q, config["max_side"], existing),
        "coco": lambda s, q: coco(s, out, q, config["max_side"], existing),
    }
    for source, settings in config["sources"].items():
        quota = settings["images"] - counts.get(source, 0)
        if quota <= 0:
            continue
        # A source with a "model" is generated under its own name; the others are fixed.
        make = makers.get(source) or (
            lambda s, q, source=source: generated(
                source, s, out, q, config["max_side"], rng, existing
            )
        )
        with manifest.open("a") as handle:
            for line in make(settings, quota):
                handle.write(json.dumps(line) + "\n")
                handle.flush()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(json.loads(args.config.read_text()), args.out))


if __name__ == "__main__":
    main()

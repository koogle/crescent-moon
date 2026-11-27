# Crescent Moon

[![Demo on Huggingface](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/jfrick/crescent-moon) live
detection on a camera or a video file, with the 4-bit version running in the browser and streamed video support.

**Fast object detection from a small vision-language model via slots.**

Small vision-language models (VLMs) detect objects by decoding coordinates one token at
a time, with one prompt per category, so latency grows with both the number of
categories and the number of objects. We post-train a *frozen* 1.9B-parameter
Moondream 2 with slot heads appended after the image tokens. They are filled in a single
forward pass and read through the model's own coordinate, size and vocabulary heads.

On 500 COCO val2017 images the head reaches F1 0.707 at IoU 0.5 in 0.09 s per image
(Moondream 3.1, 9B: 0.700 in 0.21 s), and 0.718 when it also reads a 2×2 grid of tiles,
which brings its recall of objects smaller than one feature-grid cell to Moondream's
level. On Pascal VOC 2007, which none of the models was tuned on, it scores F1 0.858
against Moondream 3.1's 0.857. With all 80 categories requested it needs about 59× less
time per image, which makes continuous detection on streamed video practical. An
auxiliary masked-diffusion denoising loss, category prompts, size-weighted losses and
teacher-only labels do not help; we report what did and did not work.

- Results: [`docs/results.md`](docs/results.md) (overview) and
  [`docs/training-log.md`](docs/training-log.md) (every run, negative results included).
- In the browser: [`docs/webgpu.md`](docs/webgpu.md).
- Paper: [`paper/main.tex`](paper/main.tex), two pages; editable copy as a
  [Claude Doc](https://claude.ai/artifact/SWvyTurEodiG2uWnKKQaWc).
- Weights: HF model repository
  [`jfrick/crescent-moon`](https://huggingface.co/jfrick/crescent-moon) (slot heads and,
  under `web/`, the 4-bit browser export).
- Run outputs: HF dataset
  [`jfrick/lm-exploration-results`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-167/post-training/detection),
  folder `pr-167/post-training/detection/<run id>/`.

## Setup

From the repository root, with [uv](https://docs.astral.sh/uv/):

```sh
uv sync --extra dev --extra modal     # add --extra model for PyTorch
git config core.hooksPath .githooks   # ruff before each commit
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Training and evaluation run on [Modal](https://modal.com) and need a Modal token and a
Hugging Face token in the environment; the demo is a Hugging Face Space.

## Inference

On one machine (the `model` extra; a GPU, or slowly on a CPU):

```python
from huggingface_hub import hf_hub_download
from PIL import Image

from crescent_moon.train_slots import SlotDetector

head = hf_hub_download("jfrick/crescent-moon", "slots-teacher-coco-rank64/head.pt")
detector = SlotDetector(head, "cuda", threshold=0.29, tiles=1)  # tiles=2 also reads a 2 x 2 grid
result = detector.detect(Image.open("photo.jpg"), ["person", "dog", "car"])
result["boxes"], result["scores"]  # per phrase: [x0, y0, x1, y1] in pixels, and scores
```

Phrases are COCO category names. `space/demo.json` lists the published heads with their
thresholds.

## Training

A head is one config in `configs/` and one Modal call (an H100, about three hours for
20,000 steps). The backbone stays frozen; image features are encoded once and cached on
the volume.

```sh
uv run modal run --detach modal_app.py::slots --run-id slots-teacher-coco-rank64 \
  --config configs/slots-teacher-coco-rank64.json
```

The head lands on the Modal volume `lm-exploration-parallel-detection` under
`slots/<run id>/`. Configs that train on teacher labels read the private datasets
`jfrick/crescent-moon-teacher-labels-gemini-4-*`; `configs/slots-rank64.json` needs COCO
only.

## Evaluation

```sh
# A trained head on the 500 COCO test images, then Pascal VOC.
uv run modal run --detach modal_app.py::main --run-id RUN-coco \
  --config configs/slots-coco.json --slots-run RUN --gpu L40S
uv run modal run --detach modal_app.py::main --run-id RUN-voc \
  --config configs/voc-slots.json --slots-run RUN --gpu L40S

# The baselines: Moondream 2 through Transformers, Moondream 3.1 through Photon.
uv run modal run --detach modal_app.py::main --run-id moondream2-001
uv run modal run --detach modal_app.py::photon --run-id moondream31-001 \
  --config configs/moondream3.1-photon.json

# Recompute scores from saved predictions.
uv run python -m crescent_moon.run score --predictions artifacts/RUN-coco/predictions.jsonl
```

`--threshold`, `--tiles`, `--max-crops` and `--weight-bits` override the head's settings;
`--all-slots --threshold 0 --tiles 2` records every candidate, which
`scripts/study_selection.py` uses to choose the threshold on `configs/slots-coco-tune.json`.
Results land on the volume under `evaluations/<run id>/` and the launching command saves
the summary under `artifacts/<run id>/`.

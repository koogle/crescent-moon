# Results overview

The measured results in one place. Each table names the section of the
[training log](training-log.md) that holds the runs, the protocol and the full tables.
All scores are at IoU 0.5 unless stated; COCO is a seeded sample of 500 val2017 images
(1,446 phrases), Pascal VOC a seeded 1,000 images of VOC 2007 test, and only categories
present in an image are prompted. Slot-head thresholds are chosen on 500 further COCO
val images.

## Against Moondream

One L40S per model; Moondream 2 and 3.1 run through Photon.

| Model | COCO F1 | COCO P / R | COCO F1, IoU 0.75 | VOC F1 | Seconds per COCO image |
|---|---|---|---|---|---|
| Moondream 2 `detect` (1.9B) | 0.586 | 0.501 / 0.706 | | 0.851 | 0.220 |
| Moondream 3.1 (9B) | 0.700 | 0.682 / 0.719 | 0.560 | 0.857 | 0.208 |
| Slot head `slots-teacher-coco-rank128`, fast | 0.707 | | 0.569 | 0.858 | 0.09 |
| Same head, 2 x 2 tiles | 0.720 | | 0.580 | 0.846 | 0.229 |
| Slot head `slots-teacher-coco-rank64`, fast | 0.705 | 0.779 / 0.644 | 0.555 | 0.840 | 0.078 |

Log: "Baseline results", "Pascal VOC 2007 test", "Small objects, tiles and duplicates",
"Training for small objects, without COCO, and without denoising". The baselines do not
reproduce Moondream's published figures ("Against Moondream's published numbers").

Recall on the 500 COCO images by object size in cells of the 27 x 27 feature grid:

| Object size | Moondream 2 | Moondream 3.1 | Rank-128 head, fast (before fusion) | Rank-128 head, tiled |
|---|---|---|---|---|
| Under 1 cell | 0.36 | 0.36 | 0.22 | 0.37 |
| 1 to 4 cells | 0.70 | 0.72 | 0.61 | 0.70 |
| 4 cells and over | 0.87 | 0.87 | 0.88 | 0.84 |

## Speed

| Measurement | Moondream 2 | Slot head | Moondream 3.1 |
|---|---|---|---|
| One image, 6 categories, through the demo page | 0.343 s | 0.081 s | 0.246 s |
| Video frame, all 80 categories | | 0.047 s | 2.79 s |
| MOT17 frame, "person" only | 0.260 s | 0.104 s | 0.206 s |

Stages of the fast head on an L40S (`slots-bins-16`, 0.078 s): vision encoder 0.046 s,
image prefix 0.016 s, slot pass 0.016 s. Custom CUDA kernels for Moondream 2's sequential
decoder cut its cost per box by 1.85x against eager PyTorch on an L4, without reproducing
eager's boxes exactly. Log: "Demo", "Video: MOT17", "Where the time goes", "Custom decode
kernels".

## Video (MOT17, every third frame of two sequences)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Moondream 2 | 0.638 | 0.308 | 0.415 |
| Moondream 3.1 | 0.816 | 0.291 | 0.429 |
| Rank-128 head, fast | 0.652 | 0.341 | 0.447 |
| Rank-128 head, tiled | 0.559 | 0.446 | 0.496 |
| `slots-tilecrops-teacher-coco-rank64`, tiled, tracked | | | 0.506 |

The tracker cuts the share of people lost in the next frame by a third with F1
unchanged. Log: "Video: MOT17".

## For the device

`slots-teacher-coco-rank64`, fast, COCO:

| Setting | F1 | Seconds per image, L40S |
|---|---|---|
| BF16 weights, all crops | 0.705 | 0.078 |
| 8-bit weights | 0.705 | |
| 4-bit weights, head trained on BF16 | 0.660 | |
| One crop, BF16 weights | 0.684 | 0.046 |
| 4-bit weights, head trained on 4 bits | 0.675 | |
| 4-bit weights and one crop (what the browser runs), any of the three heads | 0.654 to 0.655 | |

The browser's forward pass on the 4-bit export matches PyTorch on three images: every
activation within 3.2e-4 of the largest value, every box bin, label token and selected
box identical, also on a real GPU. In Chromium on a Modal L4 the pass takes 0.75 s per
frame (vision 0.12 s, decoder 0.59 s), from 2.46 s with the first kernels. On a GitHub
macOS runner's Apple GPU it takes 2.2 s, where the kernels tuned on the L4 took 13.9 s;
the page times the kernel choices on each device. A phone is not measured. Log: "Crops for the
current head", "Low-bit backbone weights", "Heads trained on the 4-bit backbone"; [`webgpu.md`](webgpu.md).

## What moved the result

| Change | Effect on COCO F1 | Log section |
|---|---|---|
| Boxes read through Moondream's coordinate bins instead of regression | 0.43 to 0.56 | Training runs |
| 45k images and 20,000 steps instead of 11k and 6,000 | 0.56 to 0.62 | Training runs |
| Small objects kept in training | 0.622 to 0.636, small-object recall 0.10 to 0.26 | Training runs |
| Adapter rank 64 instead of 16 | 0.636 to 0.657 | Loss and capacity ablations |
| 18k Gemini 4 Carbon-labelled images added to COCO | 0.657 to 0.673 | Training on Gemini 4 Carbon labels |
| Combined score, duplicates removed | 0.673 to 0.694 | Scoring slots |
| Fusion at IoU 0.5, box confidence^0.25 | 0.694 to 0.705 | Small objects, tiles and duplicates |
| 2 x 2 tiles at inference | 0.705 to 0.714 (rank 128: 0.707 to 0.720) | Small objects, tiles and duplicates |

## What did not help

| Change | Outcome | Log section |
|---|---|---|
| Size-specialised slot groups | F1 0.549 against 0.590 | Training runs |
| Objectness trained toward box IoU | Tighter boxes, fewer small objects, F1 0.631 against 0.636 | Loss and capacity ablations |
| "Nothing here" only near labelled objects | Precision falls 18 points | Loss and capacity ablations |
| Adapter rank 128 instead of 64 | F1 +0.6 points, curve area unchanged | Adapter rank 128 |
| Self-training (ignoring confident unmatched slots) | Slightly lower on every measure | Self-training |
| 20.8k more images labelled by Gemini 4 Argon | F1 0.684 against 0.694; Argon labels fewer small objects | More teacher-labelled images |
| Teacher labels without COCO | 0.681 against 0.694 for COCO alone | Training for small objects, without COCO, and without denoising |
| Category prompts, size-weighted loss | Within noise | same |
| The denoising (masked-diffusion) pass | Same scores without it, training 2.4 instead of 3.2 hours | same |
| Carbon labels instead of, or merged with, COCO's on the same images | 0.678 against 0.692 | Label source on the same images |
| Training the head on the 4-bit backbone | Recovers 1.5 of 4.5 points on COCO, loses 2 on VOC; equal in the browser's configuration | Heads trained on the 4-bit backbone |
| Image prefix pooled 2 x 2 for a faster browser frame (0.51 s against 0.80 s) | COCO F1 0.487 against 0.656; 0.565 with adapters at the image positions | A pooled image prefix |
| Tile crops in training | Small-object recall unchanged; higher precision, best VOC with tiles | Label source on the same images |

## Open work

- Recall on medium and large objects: more human-labelled images and adapting more of
  the backbone.
- Speed: train the head on single-crop features; fold a slot's box and label into fewer
  positions; time the vision encoder under the custom kernels.
- Prompted detection (a slot points at a phrase in the prompt), verified-absent
  categories in training, and labels with several valid wordings as the real test of
  one-pass diffusion.
- RefCOCO-M and LVIS for the slot head; it is evaluated on COCO and Pascal VOC only.
- Kernels: an FP32 PyTorch reference check and timing on the GPU Photon was measured on.
- Objects365 as training data: `jxu124/objects365` holds annotations only, so its images
  need a prepared subset.

# The slot detector in the browser on hand-written WebGPU kernels

Goal: run the one-pass slot detector of [this repository](../README.md) on the device,
for live camera detection in a phone's browser (target: iPhone 17 Pro, Safari, WebGPU)
without a server round trip. Runtime libraries (ONNX Runtime Web, Transformers.js) are
not used: the model is one fixed forward pass of large batched matrix products, and
hand-written WGSL kernels for exactly that pass are the point of the work.

## Status

| Milestone | State |
|---|---|
| 0. Model choice for the device | Measured ([training log](training-log.md), "Crops for the current head", "Heads trained on the 4-bit backbone"): heads trained on the 4-bit backbone, with all crops and with one, do not beat the head trained on BF16 in the browser's configuration (4 bits, one crop: COCO F1 0.654 to 0.655 for all three), so the export keeps `slots-teacher-coco-rank64`. A head on a pooled image prefix (`prefix_pool` 2: 196 image positions instead of 729) is training |
| 1. Export and reference | Implemented: `scripts/export_web.py`; the shards unpack to the model's weights bit for bit |
| 2. Kernels with unit tests | Implemented: `web/kernels.js`, 14 cases in `web/tests/kernels.test.js` pass in headless Chromium |
| 3. Full forward pass | Implemented: `web/model.js`, `web/select.js`. On three reference images every activation, bin, token and box matches PyTorch (below). The 500-image COCO comparison is not done |
| 4. On the device | The Camera tab's Model list has an entry "On this device" that runs the browser's detector; choosing it shows the weight download as its own step (a button with a progress bar), and the camera starts once the detector is loaded. A browser without WebGPU sees the entry disabled. It loads the model in headless Chromium; speed on a phone or a desktop GPU is not measured here |
| 5. Speed | Measured on a Modal L4 through Chromium's WebGPU: 0.80 s per frame after three rounds of kernel tuning, from 2.46 s ("Speed" below). FP16 shader arithmetic and a phone are not measured |

## Decisions taken

- **JavaScript + WGSL with no build step.** `web/` is three ES modules the page imports.
- **Weights come from the demo's Modal app**, which serves the export from the volume
  (`web/<export name>/`) under `/weights/`; nothing is published for it. The browser
  keeps the shards in Cache Storage, keyed by content, so they download once.
- **Code layout**: runtime in `web/`, packing in `src/crescent_moon/pack.py`, export in
  `scripts/export_web.py`, the switch in `demo_camera.html`.
- **One crop.** The device runs `max_crops` 1, for which Moondream's local crop equals
  the global crop: the vision encoder runs once and its features are used twice.
- **No key/value cache.** Prefix and slot positions go through each decoder block
  together (1,018 rows): prefix rows attend to the prefix, slot rows to everything, and
  adapters add to slot rows only. This is the same arithmetic as the Python pass with
  its prefix cache and keeps one block's keys and values in memory instead of 24.
- **FP32 arithmetic in the shaders**, with weights decoded from 4-bit fields and FP16
  scales as they are read. `shader-f16` is tuning work for milestone 5.

## Implementation

`scripts/export_web.py` rounds every Linear the detector uses (`--bits`, per component
with `--component-bits`) with one FP16 scale per group of 32 inputs, packs integers and
scales as `src/crescent_moon/pack.py` describes, stores the head's adapters and the small
parameters in FP16, and writes shards of at most 256 MB with a manifest (place and
SHA-256 of every blob, the vocabulary, the selection settings). The exported 4-bit
model is 1.2 GB. It then runs the PyTorch pass in float32 on exactly the stored weights
for each `--images` file and writes the crop, the vision features, the projected
embeddings, the prefix hidden state after blocks 0, 11 and 23, the slots' final hidden
state, every slot's outputs and the selected boxes.

`web/kernels.js` holds seven kernels: the quantised matrix product (four rows per
invocation, each weight decoded once; optional add-into and GELU), the FP16 matrix
product for adapters and objectness, layer norm, GELU, rotary embedding, attention with
an online softmax over a fused qkv matrix, and argmax with its softmax probability.
`web/model.js` loads an export into GPU buffers and builds the pass as a fixed list of
dispatches; `web/select.js` decodes bins, scores slots and fuses duplicates as
`merge.select` does for one view.

Tests run in headless Chromium, whose WebGPU here is SwiftShader on the CPU:

```sh
node web/tests/run.cjs kernels
node web/tests/run.cjs reference --export artifacts/web/NAME --timeout 300
```

`kernels` checks each kernel against plain JavaScript arithmetic. `block` runs the first
decoder block from the reference's projected embeddings: the prefix hidden state matches
PyTorch's to 3.2e-7 of its largest value on image `000000037777`. `reference` runs the
whole pass and compares every stored activation, every slot's outputs and the boxes; on
SwiftShader it takes about 33 minutes per image. `steps`
prints each buffer's statistics after every step, for locating a divergence.

Measured with `reference` on three COCO val2017 images, the 4-bit export of
`slots-teacher-coco-rank64`, FP32 shaders on SwiftShader. Differences are the largest
absolute difference as a share of the reference's largest magnitude:

| Compared with PyTorch | `000000037777` | `000000006763` | `000000009769` |
|---|---|---|---|
| Vision encoder output | 2.0e-5 | 2.4e-5 | 2.4e-5 |
| Projected image embeddings | 2.6e-5 | 3.3e-5 | 2.9e-5 |
| Prefix hidden state after decoder block 0 | 6.0e-6 | 8.1e-6 | 7.6e-6 |
| Prefix hidden state after decoder block 11 | 1.6e-6 | 1.2e-6 | 2.8e-6 |
| Prefix hidden state after decoder block 23 | 9.7e-6 | 1.8e-5 | 1.9e-5 |
| Slots' final hidden state | 1.3e-4 | 2.0e-4 | 3.2e-4 |
| Objectness logits | 1.5e-4 | 8.9e-5 | 2.1e-4 |
| Box bins (48 slots x 4) identical | 192 of 192 | 192 of 192 | 192 of 192 |
| Label tokens (48 slots x 3) identical | 144 of 144 | 144 of 144 | 144 of 144 |
| Bin probabilities | 1.1e-4 | 2.3e-4 | 1.6e-4 |
| Token probabilities | 6.1e-4 | 2.5e-4 | 4.2e-4 |
| Selected boxes (label, corners to 0.1 pixel, score to 0.001) identical | 15 of 15 | 7 of 7 | 4 of 4 |

Found on the way: `tanh` returned NaN for large arguments on SwiftShader, which broke
the first MLP of the vision encoder; GELU now clamps the argument to [-10, 10], where
FP32's tanh is already 1.

## Speed

`modal_app.py::browser` runs suites of `web/tests/` in Chromium on a Modal GPU through
Vulkan, against an export on the volume; `bench` times the pass per stage over several
frames and, with GPU timestamp queries, per kernel:

```sh
uv run modal run modal_app.py::browser --run-id RUN --export slots-teacher-coco-rank64-w4 \
  --suites bench,reference
```

One L4 (NVIDIA, FP32 shaders; the adapter does not offer `shader-f16`), the 4-bit export,
median of warm frames, seconds per frame (runs `web-bench-*` on the volume under
`evaluations/`):

| Kernels | Vision | Decoder | Total |
|---|---|---|---|
| First version: scalar loads, 4 rows per invocation of the matrix product | 0.473 | 1.910 | 2.459 |
| vec4 loads and dot products, 8 rows per invocation | 0.252 | 0.960 | 1.259 |
| A tile of 4 outputs x 16 rows per invocation | 0.171 | 0.662 | 0.868 |
| The same, and attention sharing key and value loads between 2 query rows | 0.166 | 0.600 | 0.801 |
| The same, and layer norm on vec4s | 0.152 | 0.598 | 0.783 |

Other tiles of the matrix product (outputs x rows): 4 x 8 0.983, 8 x 8 0.907, 2 x 16 1.049,
8 x 16 1.116 s. Attention with 4 or 8 query rows per invocation is slower than with one
(1.40 and 1.36 s): its per-row state no longer fits in registers.

GPU seconds per kernel in the fourth row: the decoder's quantised products 0.343, its
attention 0.181, adapters 0.042, layer norm 0.023; the vision encoder's products 0.075,
attention 0.057, layer norm 0.028. The decoder's products are 2.5 TFLOP, so they run at
about 7 TFLOPS, close to what an L4 does in FP32; what is left to gain on this GPU is
attention and layer norm, about 0.2 s. Beyond that the frame needs less arithmetic (fewer
image positions in the prefix, which means training a head for it) or cheaper arithmetic
(FP16 where a GPU is faster with it).

**Tuning on the device.** The best tile depends on the GPU, so the runtime measures it:
`tune` in `web/model.js` times each of the kernels' `CHOICES` (six tiles of the quantised
product, attention with 1, 2 or 4 query rows) on the work of one decoder block when the
model loads, builds the pass with the fastest, and `loadModel` keeps the choice in the
browser's local storage. The camera page shows the choice under the Model list. On the
L4 it takes 5.4 s and picks 4 x 16 and 2 query rows (run `web-bench-tune-003`; seconds
per product: 4 x 16 0.0043, 8 x 8 0.0044, 4 x 8 0.0049, 2 x 16 0.0053, 2 x 8 0.0057,
4 x 4 0.0059; per attention: 2 rows 0.0077, 1 row 0.0106, 4 rows 0.0194). Every choice is
checked by `web/tests/kernels.test.js`. No measurement on a phone yet.

With these kernels the pass matches the PyTorch reference on the L4 on all three images
(36 of 36 checks; run `web-bench-final-l4`, and `web-bench-norm4-001` with the vec4 layer
norm, which takes 0.011 GPU seconds per frame in place of 0.051).

Tried on attention after that, each measured on the L4 and none faster than the 0.80 s
above: straight-line code over named vec4s instead of local arrays (0.79 s with 2 query
rows, 0.96 with 4, 0.88 with 1), keys and values regrouped by head so that a head reads
them contiguously (0.82 s; the regrouping itself takes 2 ms), and a loop without the
rescaling branch (0.80 s). Attention is bound by arithmetic, not by memory layout or the
branch.

**Pooled image prefix.** A head's `prefix_pool` averages the 27 x 27 grid of image
embeddings over 2 x 2 cells before the decoder (196 image positions instead of 729); the
runtime does it with the `pool-grid` kernel and reads the prefix length from the
manifest. Timed with the current head's weights exported under `prefix_pool` 2 (run
`web-bench-pool2-l4`; that head was not trained for it, so this measures speed only):

| Prefix | Vision | Decoder | Total |
|---|---|---|---|
| 729 image positions | 0.166 | 0.600 | 0.801 |
| 196 image positions | 0.172 | 0.303 | 0.510 |

Against PyTorch with the same pooling, on one image, 11 of 12 checks pass; the twelfth
finds 191 of 192 box bins identical. `slots-teacher-coco-rank64-w4-crops1-pool2`, a head
trained on the pooled prefix, reaches COCO F1 0.487 against 0.656 with the full prefix
([training log](training-log.md#a-pooled-image-prefix)), so the export keeps the full
prefix.

## Not done yet

- The 500-image COCO comparison of milestone 3 (browser F1 within 0.005 of PyTorch at the
  same rounding): it needs a real GPU, which `modal_app.py::browser` now provides.
- Timings on the iPhone 17 Pro and a desktop browser, and whether Safari's memory limit
  admits the 1.2 GB export (plus about 0.25 GB of activations and decoded parameters).
- Stage times on the camera page are wall-clock, with the GPU queue drained after each
  stage; the benchmark uses timestamp queries.
- Tracking on the device: the page's "Track across frames" applies to the server path.
- The camera frame is resized by the browser's canvas, the reference crop by libvips;
  the effect on accuracy is unmeasured.

## What runs per frame

Moondream 2 (`vikhyatk/moondream2`, revision `2025-06-21`, Apache-2.0), frozen, plus
the trained slot head (`slots-teacher-coco-rank64`, the demo's default; HF archive
`jfrick/lm-exploration-results`, `pr-167/post-training/detection/`). Source of truth for
shapes and order of operations: the model's Hub code (`config.py`, `vision.py`,
`text.py`, `region.py`, `rope.py`, `layers.py`, `image_crops.py`) and
`post-training/detection/src/crescent_moon/slots.py` and `train_slots.py`
(`SlotDetector._encode`, `_decode`, `image_cache`, `sample`).

1. **Crops.** `vision.prepare_crops`: a global 378 x 378 crop plus local crops (up to
   `max_crops`, default 12), normalised; each crop is 27 x 27 patches of 14 x 14 x 3.
2. **Vision encoder** (per crop, 729 positions): patch embedding (Linear 588 -> 1152) +
   position embedding, 27 pre-norm blocks (LayerNorm, attention with 16 heads of 72,
   LayerNorm, MLP 1152 -> 4304 -> 1152 with tanh GELU), final LayerNorm.
3. **Projection**: local crops reassembled (`reconstruct_from_crops`) and average-pooled
   to 27 x 27, concatenated with the global crop's features (2304) and passed through an
   MLP 2304 -> 8192 -> 2048: 729 image embeddings.
4. **Image prefix** (730 positions: BOS embedding + 729 image embeddings) through the 24
   frozen decoder blocks, every position attending to every other. Block: one LayerNorm,
   then attention and MLP in parallel, both added to the residual. Attention: 32 heads of
   64, fused qkv 2048 -> 6144, rotary embedding on the first 32 dims of each head
   (non-interleaved, theta 10,000), output 2048 -> 2048. MLP: 2048 -> 8192 -> 2048, tanh
   GELU. Keep each block's keys and values (the prefix cache).
5. **Slots** (48 slots x 6 positions = 288, positions 730 to 1017): input = mask + role +
   slot query embeddings (learned). Through the same 24 blocks, attending to the prefix
   cache and to all slot positions, with low-rank adapters (rank 64, kept in higher
   precision) added to qkv, proj, fc1 and fc2 at slot positions only.
6. **Read-out**: positions 0-1 of each slot through the region coordinate decoder
   (2048 -> 8192 -> 1024 bins), position 2 through the size decoder (2048 -> 8192 ->
   2 x 1024 bins), positions 3-5 through the final LayerNorm and the vocabulary head
   (2048 -> 51,200; softmax over the full vocabulary for confidences), position 0 also
   through the objectness linear. Per position keep the argmax and its probability.
7. **Selection** (CPU, JavaScript port of `merge.select`): score = objectness x label
   confidence x box confidence^0.25, threshold 0.29 (tuned on COCO), weighted box fusion
   at IoU 0.5, label tokens to category names via the checkpoint's `vocabulary`.

Not needed on the device: the token embedding table (one BOS row only), the region
encoders (used only to reveal or hint positions), tiling (2 x 2 tiles cost 5 passes).

## Budget

| Part | Weights | Work per frame, one crop |
|---|---|---|
| Vision encoder | 0.41B | 0.6 TFLOP per crop |
| Projection | 0.04B | small |
| Decoder, image prefix | 1.21B (shared) | 1.8 TFLOP + attention |
| Decoder, slots | (same) + 0.05B adapters | 0.7 TFLOP |
| Heads (region, vocabulary) | 0.16B | under 0.1 TFLOP |
| Total | about 1.87B | about 3.2 TFLOP with one crop |

Downloads at one scale (FP16) per block of 32: about 1.0 GB at 4 bits, 1.9 GB at 8 bits,
plus 0.1 GB of FP16 adapters. Activations are small (prefix cache: 24 x 730 x 2048 x 2
FP16 = 144 MB).

Speed is unmeasured. WebGPU cannot use the A19 Pro GPU's per-core matrix units, so a
first guess is 2-4 TFLOPS of FP16 shader arithmetic: about 1 s per frame at first,
perhaps 0.5 s tuned. The image prefix through the decoder is over half the work and
does not shrink with fewer slots.

## Facts that set constraints

- **Weight precision** (measured, [training log](training-log.md), "Low-bit backbone
  weights"): rounding every backbone Linear to 8 bits per block of 32 costs nothing
  (COCO F1 0.705 -> 0.705); 4 bits costs 4.5 points (0.705 -> 0.660, area 0.657 ->
  0.597; VOC unchanged). `quantize.round_weights_` is that rounding.
- **Safari memory**: one developer reports a 2,048 MB limit for Safari's web content
  process on iPhone 17 Pro ([forum thread](https://developer.apple.com/forums/thread/823061));
  not an Apple specification. iOS kills pages under memory pressure
  ([WebKit bug 303203](https://bugs.webkit.org/show_bug.cgi?id=303203)). 8-bit weights
  (about 2 GB) are therefore at or over the limit; 4 bits (about 1 GB) fit.
- **One crop**: limiting the vision encoder to one crop cut a 0.078 s frame to 0.046 s
  on an L40S and cost 1.5 points of COCO F1 for an older head (`slots-bins-16`); not yet
  measured for the current head.

## Milestones as planned

0. **Model choice for the device** (Python, Modal).
   Close the 4-bit gap and fix the configuration the browser will run:
   - train the slot head on the 4-bit-rounded backbone (its adapters see the rounded
     weights; a `weight_bits` option in training through `round_weights_`);
   - if that is not enough: keep the most sensitive layers at 8 bits (measure per
     component: vision, decoder, heads), or better 4-bit rounding;
   - measure one crop and `max_crops` 1-2 for the current head.
   Done when one configuration (bits per component, crops) is chosen with its COCO and
   VOC numbers in the training log.
1. **Export and reference** (Python). A script in `scripts/`
   that writes, for the chosen head and configuration:
   - packed weights: per tensor, signed integers row-major [out, in] packed into u32
     words, one FP16 scale per block of 32 along the input; adapters, norms, biases,
     queries, mask, roles in FP16; shards of at most 256 MB; a JSON manifest (name,
     shape, format, offset, byte length, SHA-256) and the label vocabulary;
   - reference activations on a few fixed images (crops, vision output, projection,
     prefix hidden state after blocks 0, 11 and 23, slot hidden state, objectness, bin
     and label argmaxes and probabilities, final boxes), computed with exactly the
     dequantised weights the packing holds.
   Published to HF, not Git. Done when unpacking the shards in Python reproduces the
   rounded weights bit for bit.
2. **Kernels with unit tests** (WGSL + JavaScript, `web/`): quantised matrix
   product with fused epilogues (bias, GELU, residual add) for M = 729, 730, 288 and the
   small read-out sizes; LayerNorm; bidirectional attention with online softmax (head
   dims 72 and 64, keys up to 1,018, prefix cache); partial rotary embedding; adapter
   products; patch extraction and normalisation; argmax/probability reductions over
   1,024 and 51,200. Each against a CPU reference in tests. FP16 shader arithmetic where
   the device offers `shader-f16`, FP32 otherwise.
3. **Full forward pass**: weight loading (streamed, cached in the Cache Storage API or
   OPFS so 1 GB downloads once), the pass of "What runs per frame", the JavaScript
   selection. Done when every reference activation matches within tolerance and the
   boxes on the fixed images match the reference; then on 500 COCO val images the
   browser's F1 is within 0.005 of the PyTorch evaluation at the same rounding.
4. **On the device**: a "Run on this device" switch in the Camera page, per-stage GPU
   timings shown on the page (timestamp queries where available), measured on the
   iPhone 17 Pro and a desktop browser.
5. **Speed**: larger tiles and subgroup operations in the matrix product, fusing
   LayerNorm and rotary into neighbours, reusing buffers; re-measure after each step.

## Testing environment

The cloud session has Chromium with Playwright. First task of milestone 2: confirm
WebGPU works in headless Chromium there (software adapter), otherwise use Dawn's
Node.js bindings for kernel tests. Correctness is tested there; speed only on real
devices, which the user runs.

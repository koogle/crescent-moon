## Crescent Moon

Fast object detection from a small vision-language model: a frozen Moondream 2 with a
trained slot head that returns every category in one forward pass
([code and paper](https://github.com/koogle/crescent-moon),
[weights](https://huggingface.co/jfrick/crescent-moon)).

**Camera** detects live on a camera or a video file; its first model runs in your
browser on WebGPU with 4-bit weights (one download of about 1 GB), the others on this
Space's GPU. **Images** compares Moondream 2's own `detect` with the slot heads, and
**Video** runs one model over a clip. Categories are comma-separated COCO names; empty
means all 80. Moondream 2 answers one category per pass, so its time grows with the
number of categories; the slot heads answer all of them in one pass.

Slot-head boxes show their score: objectness × label confidence × (box confidence)^0.25,
a ranking score rather than a probability. "Fast" heads read the full image once;
"tiled" also reads a 2 × 2 grid of tiles for small objects, at about 2.5 times the time.

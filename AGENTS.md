# Crescent Moon

Fast object detection from a small vision-language model: a frozen Moondream 2 with a
trained slot head that returns every category in one forward pass. Read `README.md`
(method, every measured table, run commands) before changing code, and `docs/webgpu.md`
before touching the on-device work.

- `src/crescent_moon/`: the package. `slots.py` (head, loss), `train_slots.py`
  (training, `SlotDetector` inference with tiling), `merge.py` (selection), `run.py`
  (evaluation harness), `tasks.py` (datasets), `metrics.py`, `tracking.py`,
  `quantize.py`, `teacher.py` / `images.py` (OpenRouter labelling), `moondream.py`,
  `photon.py`, `kernel.py` + `decode_step.cu`, `demo.py`, `deadline.py`, `pack.py` (the
  packed weight format the browser reads).
- `web/`: the detector in the browser, plain JavaScript and WGSL with no build step:
  `kernels.js` (WGSL kernels), `model.js` (loader, forward pass), `select.js`
  (selection), `tests/` (run with `node web/tests/run.cjs kernels`). The Python pass in
  `slots.py` / `train_slots.py` is the reference; `scripts/export_web.py` writes the
  weights and reference activations the browser tests compare against.
- `modal_app.py`: Modal app `crescent-moon` (training, evaluation, Photon, deadline
  supervisor). `demo_app.py` + `demo_camera.html`: Modal app `crescent-moon-demo`.
- `configs/`: run configs. `scripts/`: studies, figures, label tools. `tests/`.
  `paper/`: LaTeX paper. `docs/`: plans and findings.

Run commands from the repository root. Environment: `uv sync --extra dev --extra modal`
(add `--extra model` for PyTorch), then `uv run pytest`, `uv run ruff check .` and
`uv run ruff format --check .`. Enable the pre-commit hook once per checkout with
`git config core.hooksPath .githooks`.

## Outside Git

- Modal volume `lm-exploration-parallel-detection` (name kept from the project's first
  home): trained heads under `slots/<run id>/`, evaluations, teacher images, HF cache.
  Configs name paths on it as `/runs/...`.
- HF model repository `jfrick/crescent-moon` (public): the demo's slot heads and, under
  `web/`, the browser exports. An export is also on the volume under `web/<name>/`.
- HF dataset `jfrick/lm-exploration-results`, folder `pr-167/post-training/detection/`:
  every run's outputs, head snapshots, selection studies and figures.
- Private HF datasets `jfrick/crescent-moon-teacher-labels-gemini-4-carbon` and
  `-argon`: teacher labels and their images.
- Credentials come from the environment: HF and Modal tokens, and the OpenRouter key in
  `OPENROUNTER_TOKEN` (spelled that way). The OpenRouter key is never put in a Modal
  secret. Never commit credentials or checkpoints.

## Writing

Distinguish mockups, implemented behaviour and measured results. Describe what code,
docs and pull requests do; do not state what was removed or is not used (that history
belongs in commit messages). Preserve experiment provenance and negative results: a
run that did not help keeps its table and a sentence on why.

## No backward compatibility

This repository only moves forward. Code serves the current configs, architectures and
data formats. Do not add or keep compatibility paths for old checkpoints, result
schemas or launchers (legacy loaders, format fallbacks, version switches). Old artifacts
stay on HF with their provenance; if one is needed later, write the compatibility code
then, for that one use. Safety and correctness checks against the current format stay.

## Reuse existing infrastructure

One training loop (`train_slots.train`), one evaluation harness (`run.evaluate` with
`metrics.summarize`), one launch pattern (`modal_app.launch` with `enforce_deadline`),
one selection (`merge.select`). A new model, dataset, loss or study extends these (a
parameter, a config key, a dataset kind in `tasks.py`); it does not get its own loop,
harness, launcher or metric code. If the existing piece cannot serve the new case,
extend it and say in the pull request why. A parallel implementation needs a stated
reason, and the old one is consolidated into it in the same change. Speed problems in
shared code are fixed in that code. Pull requests are split by meaning: code changes in
reviewable units, findings (plans, measured results) in their own documentation change.

## Modal job runtime

Every Modal job launched by an agent must stop automatically within 24 hours (86,400
seconds) of launch, including detached and background jobs and the deployed demo.
Enforce this through Modal's API before launch so shutdown survives agent disconnection.

- Set a server-side execution timeout of at most `24 * 60 * 60` seconds with
  `@app.function(timeout=..., retries=0)`. Preserve shorter existing limits (evaluations
  one hour, training eight).
- A Function timeout covers execution time per attempt, excluding queue and start-up
  time; retries reset it. For the overall deadline from launch, fix the deadline before
  anything runs, bound each call by the remaining budget, refuse work after it, and
  spawn the independent supervisor (`enforce_deadline`, built on
  `crescent_moon.deadline`) that cancels outstanding calls through
  `modal.FunctionCall.from_id(call_id).cancel()`. Track and cancel children explicitly;
  stopping a parent does not stop them.
- Keep retries disabled. Do not restart a timed-out job automatically to bypass the cap.
- `FunctionCall.get(timeout=...)` only limits how long the caller waits. Reminders,
  local timers and manual cleanup are not shutdown enforcement.
- The demo is deployed only through `python demo_app.py`, which spawns `shutdown_at` to
  stop the app just under 24 hours later. A bare `modal deploy demo_app.py` skips that.
- Save checkpoints and partial results early enough to survive shutdown.

See Modal's [timeout semantics](https://modal.com/docs/guide/timeouts) and
[FunctionCall cancellation](https://modal.com/docs/reference/modal.FunctionCall#cancel).

## Source and artifact ownership

Git holds executable logic, configs, tests, small test fixtures, and concise plans, run
instructions and experiment summaries. Hugging Face holds experiment outputs:
checkpoints, result JSONs, predictions, metrics, figures, built PDFs, exported weights
and reference activations. Do not commit generated artifacts merely because they are
small or JSON.

- Scripts take explicit artifact paths (downloaded into the ignored `artifacts/`
  directory). Nothing requires generated results to be checked into Git.
- Docs name the HF repository and folder; no per-file artifact indexes in Git.
- Never publish credentials or private data as part of an upload.

## Reports

Building reports by concatenating strings in code (`lines += [...]`, f-string tables,
prose in `print`/`write_text`) is a red flag; avoid it. Scripts emit structured results
(JSON/CSV); render tables with a library or a checked-in template. Findings and
narrative are written by hand in `README.md` and `docs/`, not in generator code.

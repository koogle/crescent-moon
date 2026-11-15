---
name: project-documentation
description: Keep the README, docs/ (training log, results overview, plans) and the paper in step with the code and measurements while implementing.
---

1. Read `AGENTS.md`, `README.md`, `docs/training-log.md` and the plan in `docs/` that owns
   the next action.
   Check important claims against the source and the archived results.
2. Keep hypotheses, implemented behaviour and measured results separate. A number in
   the training log names its run id; the run's outputs are on HF.
3. At meaningful checkpoints record durable corrections, methods, limitations and
   measured outcomes, negative ones included, in the owning section of `docs/training-log.md`; headline numbers also go in
   `docs/results.md`, whose "Open work" holds open questions. Plans carry a status line.
   The README holds only the demo link, the abstract, run commands and development notes.
4. Update documents in place. When a number changes, update every place that quotes it
   (the README's abstract, `docs/`, `paper/main.tex`, the model card).
5. Before handing off, committing or pushing, reconcile the docs with the code and the
   checks actually run: links, `git diff --check`, `uv run pytest`, `uv run ruff check .`.
   Do not erase unresolved work or describe an experiment that did not run as done.

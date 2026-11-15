---
name: project-documentation
description: Keep the README, docs/ and the paper in step with the code and measurements while implementing.
---

1. Read `AGENTS.md`, `README.md` and the plan in `docs/` that owns the next action.
   Check important claims against the source and the archived results.
2. Keep hypotheses, implemented behaviour and measured results separate. A number in
   the README names its run id; the run's outputs are on HF.
3. At meaningful checkpoints record durable corrections, methods, limitations and
   measured outcomes, negative ones included, in the owning section. Open questions go
   in the README's "Open work"; plans carry a status line.
4. Update documents in place. When a number changes, update every place that quotes it
   (README tables, `docs/`, `paper/main.tex`, the model card).
5. Before handing off, committing or pushing, reconcile the docs with the code and the
   checks actually run: links, `git diff --check`, `uv run pytest`, `uv run ruff check .`.
   Do not erase unresolved work or describe an experiment that did not run as done.

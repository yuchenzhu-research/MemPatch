# `experiments/archive/` — preserved historical / reference logic

This directory holds **archived research code** that is no longer part of the
canonical Paper 1 evaluation path. It is kept for historical reference and as a
starting point for future experiments. It is **not** wired into the active
package and is **not** a paper-facing entrypoint.

## What lives here

- **E2 — Action ablation** (`legacy/action_ablation_eval.py`): typed-action
  vocabulary ablations.
- **E3 — Composition** (`legacy/composition_eval.py`): compositional revision
  evaluation.
- **E4/E6 — STALE / CUPMem external validation**
  (`stale_adapter.py`, `stale_cupmem_comparison.py`,
  `stale_style_retrace_validation.py`, `cupmem_adapter.py`, `cupmem_bridge.py`):
  external stale-memory baseline/compatibility analysis.
- **Older comparison / diagnostic runners and data models** (`legacy/*`,
  `methods.py`, `metrics.py`, `episodes_dev.py`, `fixtures.py`,
  `run_model_matrix_api_eval.py`, `run_stagec_adapter_eval.py`): superseded by
  the shared evaluation engine in `src/retracemem/evaluation/multiagent/`.

## Rules

1. **Not imported by `src/retracemem/`.** The active package never depends on
   anything under `experiments/archive/`. A regression test
   (`tests/test_active_package_boundary.py`) enforces this.
2. **Not a primary README command.** The canonical commands are
   `python3 scripts/evaluate.py {stage-a,stage-b,stage-c}`.
3. **Reference only.** If E2/E3/E4 results are needed for final paper numbers,
   they should be **reimplemented through the shared evaluation pipeline**
   (`retracemem.evaluation.multiagent`), not revived from this directory as-is.
4. Tests that exercise archived code live under `tests/experiments/archive/` and
   are clearly marked as archived/legacy status.

The canonical active path is **Stage A / Stage B / Stage C over the shared
ReTrace evaluation pipeline**.

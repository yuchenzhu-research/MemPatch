# Repository Layout And Artifact Policy

This repository is governed by two active tracks:

- `ReTrace-Bench`: benchmark data, scoring, baselines, leakage checks, and
  release packaging.
- `ReTrace-Learn`: method/runtime/training code for learned graph extraction and
  typed revision proposal.

## Source-Controlled Directories

- `src/`: ReTrace-Learn and deterministic authorization/runtime code.
- `benchmark/retrace_bench/`: benchmark schemas, taxonomy, scorers, baselines,
  validation, and benchmark-specific provider adapters.
- `scripts/`: reproducible entrypoints for data generation, validation,
  leakage checks, baselines, SFT export, and evaluation.
- `tests/`: regression tests. Keep these unless the corresponding code/data
  path is removed in the same change.
- `data/retrace_bench/`: ReTrace-Bench v1.0 evaluation splits only.
  - `main_3000_en/`: controlled benchmark main split (3000).
  - `hard_300_en/`: long-context / multi-evidence stress split (300).
  - `realistic_100_en/`: realistic-style workflow split (100), annotation pending.
  - `calibration_80_en/`: smoke / quickstart split (80).
- `data/retrace_learn/`: method-track fixtures, boundary-audit data, and manifests.
  Clean v1.0 training datasets will be generated under `data/retrace_learn/v1_0/`.
- `docs/`: durable project, method, benchmark, and release documentation.
- `references/`: lightweight reference registries and notes only.
- `release/huggingface/ReTrace-Bench/`: exact Hugging Face dataset release
  package. The v1.0 release exposes the `main`, `hard`, `realistic`, and
  `calibration` splits mapped to the source data under `data/retrace_bench/`.

## Local-Only Directories

The following are intentionally ignored and should not be committed:

- `outputs/`: run outputs, predictions, metrics, temporary reports.
- `artifacts/`: local generated artifacts.
- `local/`: catch-all local workspace for temporary training corpora, converted
  files, external checkouts, and machine-specific scratch material.
- `models/`, `checkpoints/`, `adapters/`: trained weights and adapters.
- `data_external/`: downloaded external benchmark/data snapshots.
- `.external_repos/`, `.reference_cache/`: local cloned external repositories
  or paper/source inspection caches.
- `analysis/`: scratch analysis notes for one-off project iteration.
- `.venv*/`, caches, `__pycache__/`, `.pytest_cache/`, `.pycache_compile/`.

## Training Placement

For server training:

- Keep committed training/export source data under
  `data/retrace_learn/v1_0/` or generated from scripts into `outputs/`.
- Write training runs to `outputs/local_training/` or a user-specified ignored
  run directory.
- Write checkpoints/adapters to `checkpoints/`, `models/`, or `adapters/`.
- Use `local/` for temporary training corpora, converted data, external clones,
  or framework-specific scratch directories that should never enter git.
- Commit only small manifests, configs, scripts, and final release metadata.
  Do not commit model weights, raw run outputs, caches, or downloaded external
  corpora.

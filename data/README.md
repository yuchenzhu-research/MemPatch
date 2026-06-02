# ReTrace Data Layout

This directory organizes datasets, validation splits, and supervision pools.

## Structure

### `data/retrace_bench/`
Contains held-out benchmark and calibration split data.
- **`test_800_templateheldout_en`**: The canonical paper-facing held-out benchmark test split containing 800 scenarios. All headline numbers come from here. No training targets are included.
- **`test_800_en`**: Old prototype/diagnostic benchmark split (kept for historical compatibility only). Must **not** be used for paper headline numbers.
- **`sample_80_hard_en`**: A small calibration / quickstart / smoke dataset for debugging and pipeline verification. On Hugging Face it may be exposed as the `validation` split for dataset-viewer compatibility only; it is **not** a model-selection / checkpoint-selection validation set.

### `data/retrace_supervision/`
Contains synthetic supervision and selection pools shared across the two active
tracks. ReTrace-Bench publishes them as HF `train` / `dev` splits, and
ReTrace-Learn consumes them for SFT / selection. They are **not** held-out
benchmark test sets and may contain training targets.
- **`train_3000_en`**: Supervision/selection pool containing 3000 scenarios.
- **`dev_400_en`**: Dev/selection pool containing 400 scenarios.

### `data/retrace_learn/`
Legacy/internal ReTrace-Learn method data (e.g. `v1/internal_dev` and `v1/boundary_audit`).
- Kept strictly for internal/legacy training diagnostics. Do not use for scoring external baselines.

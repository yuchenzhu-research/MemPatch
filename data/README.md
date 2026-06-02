# ReTrace Data Layout

This directory organizes datasets, validation splits, and supervision pools.

## Structure

### `data/retrace_bench/`
Contains held-out benchmark and calibration split data.
- **`sample_80_hard_en`**: A small calibration/quickstart dataset for debugging and diagnostics.
- **`test_800_templateheldout_en`**: The official paper-facing held-out benchmark split containing 800 scenarios. No training targets are included here.
- **`test_800_en`**: Old prototype/diagnostic benchmark split (kept for historical compatibility only).

### `data/retrace_supervision/`
Contains synthetic supervision and selection pools for the ReTrace-Learn pipeline. These are **not** held-out benchmark test sets and may contain training targets.
- **`train_3000_en`**: Supervision selection pool containing 3000 scenarios.
- **`dev_400_en`**: Dev pool containing 400 scenarios.

### `data/retrace_learn/`
Legacy/internal ReTrace-Learn method data (e.g. `v1/internal_dev` and `v1/boundary_audit`).
- Kept strictly for internal/legacy training diagnostics. Do not use for scoring external baselines.

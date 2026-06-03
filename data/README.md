# ReTrace Data Layout

This directory organizes the ReTrace-Bench v1.0 evaluation splits and the
ReTrace-Learn supervision pools.

## Structure

### `data/retrace_bench/` (ReTrace-Bench v1.0)
Four paper-facing evaluation splits (public names `main` / `hard` / `realistic`
/ `calibration`). All are de-actionalized and pass a decision-word leakage
audit; benchmark rows carry no training targets.
- **`main_3000_en`**: Controlled benchmark main split (3000 scenarios). Primary headline results.
- **`hard_300_en`**: Long-context / multi-evidence / multi-memory stress split (300 scenarios; 20–100 events, ≥5 memories, ≥2 evidence events per case).
- **`realistic_100_en`**: Realistic-style workflow split (100 scenarios). `source_type = realistic_style_synthetic`, `annotation_status = pending`; `hidden_gold` is intentionally empty until human annotation (template in `annotations_template.jsonl`). No human validation or public-source provenance is claimed.
- **`calibration_80_en`**: Smoke / quickstart split (80 scenarios). **Not** for model selection, checkpoint selection, tuning, or headline claims.

The legacy pre-v1.0 layout is recoverable from the Git tag
`legacy-retrace-bench-pre-v1.0`.

### `data/retrace_learn/supervision_*`
Synthetic supervision / selection pools for the ReTrace-Learn method track.
They are **not** benchmark test sets and may contain training targets.
- **`supervision_train_3000_en`**: Supervision/selection pool (3000 scenarios).
- **`supervision_dev_400_en`**: Dev/selection pool (400 scenarios).

### `data/retrace_learn/`
Legacy/internal ReTrace-Learn method data (e.g. `v1/internal_dev` and `v1/boundary_audit`).
- Kept strictly for internal/legacy training diagnostics. Do not use for scoring external baselines.

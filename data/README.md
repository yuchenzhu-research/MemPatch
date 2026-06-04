# ReTrace Data Layout

This directory organizes the ReTrace-Bench v1.0 evaluation splits and the
ReTrace-Learn method-track data. ReTrace-Learn uses ReTrace-Bench-derived
scenario data with declared split roles rather than a separate corpus; the
benchmark track stays method-neutral as an evaluation artifact.

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

### `data/retrace_learn/v1_0/` (legacy/internal method-track path)
`data/retrace_learn/v1_0/` is a legacy/internal path for method-track training
and validation exports. It is **not** the current canonical data policy: the
method track consumes ReTrace-Bench-derived scenarios with declared split roles,
so method training/validation data is organized from the benchmark scenario
family rather than a separate large-scale corpus. Split roles must be explicit;
do not claim leakage-free held-out evaluation if the same gold labels are used
for training. Older pre-v1.0 supervision scaffolding is recoverable only from
Git history and legacy tags.

### `data/retrace_learn/v1/` (existing internal dev / diagnostic data)
Small internal development and diagnostic datasets that already exist in the
tree (distinct from the future `v1_0/` training corpus above). They are
referenced by method configs and tests, not by the benchmark:
- **`boundary_audit/`**: method-side boundary/leakage audit rows
  (e.g. `boundary_audit_dev.jsonl`, used as exemplars by some method configs).
- **`internal_dev/`**: controlled A/B and ambiguity-scope diagnostic cases.

This is method-track development material only; it is **not** a ReTrace-Bench
split and is **not** the large-scale `v1_0/` training corpus.


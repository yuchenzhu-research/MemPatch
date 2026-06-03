# ReTrace Data Layout

This directory organizes the ReTrace-Bench v1.0 evaluation splits and the
ReTrace-Learn method-track datasets (existing internal dev data under `v1/` and
the future clean training corpus under `v1_0/`).

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

### `data/retrace_learn/v1_0/` (future clean training/validation corpus)
Clean training and validation datasets for the ReTrace-Learn method track. This
directory is the **future home** for the large-scale corpus and does not exist
yet — these will be generated natively from hidden graph, raw dialogue, and DPA
outcome pipelines (the large-scale generator is not implemented in v1):
- **`graph_sft_train/`**: Clean SFT training cases for learned Graph Extractor.
- **`graph_sft_dev/`**: Clean validation cases for learned Graph Extractor.
- **`proposer_sft_train/`**: Clean SFT training cases for learned Typed Proposer.
- **`proposer_sft_dev/`**: Clean validation cases for learned Typed Proposer.
- **`dpa_preference_train/`**: Clean SFT/DPO preference data for training.
- **`dpa_preference_dev/`**: Clean SFT/DPO preference data for validation.

Pre-v1 supervision scaffolding (`supervision_train_3000_en`, `supervision_dev_400_en`) is leaky and has been removed from the active mainline tree. It remains recoverable from Git history and legacy tags.

### `data/retrace_learn/v1/` (existing internal dev / diagnostic data)
Small internal development and diagnostic datasets that already exist in the
tree (distinct from the future `v1_0/` training corpus above). They are
referenced by method configs and tests, not by the benchmark:
- **`boundary_audit/`**: method-side boundary/leakage audit rows
  (e.g. `boundary_audit_dev.jsonl`, used as exemplars by some method configs).
- **`internal_dev/`**: controlled A/B and ambiguity-scope diagnostic cases.

This is method-track development material only; it is **not** a ReTrace-Bench
split and is **not** the large-scale `v1_0/` training corpus.


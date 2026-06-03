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

### `data/retrace_learn/v1_0/`
Clean training and validation datasets for the ReTrace-Learn method track. These will be generated natively from hidden graph, raw dialogue, and DPA outcome pipelines:
- **`graph_sft_train/`**: Clean SFT training cases for learned Graph Extractor.
- **`graph_sft_dev/`**: Clean validation cases for learned Graph Extractor.
- **`proposer_sft_train/`**: Clean SFT training cases for learned Typed Proposer.
- **`proposer_sft_dev/`**: Clean validation cases for learned Typed Proposer.
- **`dpa_preference_train/`**: Clean SFT/DPO preference data for training.
- **`dpa_preference_dev/`**: Clean SFT/DPO preference data for validation.

Pre-v1 supervision scaffolding (`supervision_train_3000_en`, `supervision_dev_400_en`) is leaky and has been removed from the active mainline tree. It remains recoverable from Git history and legacy tags.


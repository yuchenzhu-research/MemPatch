# ReTrace-Bench v1.0 — Deprecation Notice

> **ReTrace-Bench v1.0 is a legacy pilot release and is deprecated for new
> experiments. It is preserved for reproducibility of early development notes
> only. New evaluations should use the canonical ReTrace-Bench release prepared
> under v1.1.**

## What is deprecated

- **Data:** `data_legacy/retrace_bench_v1_0/**` (main/hard/realistic/calibration
  pilot splits, plus early `hard_300_en` and `realistic_100_en` pilots, and the
  held-out `private_hidden_200_en`).
- **Outputs:** `outputs/retrace_bench/v1_0/**` (early v1.0 model-suite/sanity
  outputs), retained for provenance.
- **Docs:** `docs/retrace_bench/v1_0_gated_model_pilot.md`,
  `docs/retrace_bench/v1_0_sanity_model_pilot.md`,
  `docs/retrace_bench/v1_0_sanity_error_audit.md`, and the legacy HF dataset card
  `docs/retrace_bench/dataset_card_hf.md`.

## Policy

- **Preserve, do not delete.** v1.0 stays in the repository (and in git history)
  for reproducibility of early development notes and provenance.
- **Not broken.** "Deprecated" means **superseded**, not defective. v1.0 predates
  the hardened, deterministic (seed 2027) v1.1 construction + validation
  pipeline; it should not be characterized as broken.
- **Do not mix v1.0 into canonical evaluation.** All canonical numbers, baselines,
  and paper claims come from the v1.1 splits (`data/retrace_bench_v1_1/`) and the
  HF release bundle (`hf_release/retrace_bench_v1_1/`).
- **HF archival before overwrite.** The existing Hugging Face dataset
  (`Sylvan-Vale-Moon/ReTrace-Bench`, currently the v1.0-era release) must be
  **tagged/archived** before any v1.1 upload so the legacy version is not
  silently overwritten. See `docs/retrace_bench/v1_1_hf_release_plan.md`.

## Canonical replacement

Use the canonical ReTrace-Bench release (internally "v1.1"):
- `data/retrace_bench_v1_1/` — committed canonical splits.
- `hf_release/retrace_bench_v1_1/` — HF distribution bundle (card, license,
  manifests, checksums; full JSONL rebuilt locally).
- `docs/retrace_bench/data_construction_statement.md`,
  `docs/retrace_bench/v1_1_validation_report.md`,
  `docs/retrace_bench/v1_1_hf_upload_ready_report.md`.

# ReTrace-Bench — Artifact Checklist

Status of the benchmark artifact track. `[x]` done, `[ ]` remaining.

- [x] **Hugging Face dataset card verified (v1.0).** Splits `main` (3000) /
  `hard` (300) / `realistic` (100) / `calibration` (80) load via
  `datasets.load_dataset`; JSON string columns parse with `json.loads`; card
  uses only the public split names `main` / `hard` / `realistic` /
  `calibration`, states the strong "calibration is smoke-only, not for model
  selection" semantics and the realistic `annotation_status = pending` status,
  and links back to the repository and evaluator. Package source:
  `release/huggingface/ReTrace-Bench/`, generator
  `scripts/package_hf_retrace_bench.py`.
- [x] **Official evaluator CLI.** `scripts/evaluate_retrace_bench_predictions.py`
  — no model, no API keys; `--strict/--no-strict`, `--allow-missing`,
  `--out-metrics`, `--out-scored`, `--print-table`.
- [x] **Python scoring API.** `benchmark/retrace_bench/api.py`:
  `load_scenarios`, `load_predictions`, `normalize_prediction`,
  `evaluate_predictions`; re-exports `HEADLINE_METRICS`, `AUXILIARY_METRICS`,
  `DECISIONS`, `MEMORY_STATUSES`, `FAILURE_MODES`.
- [x] **Strict validation contract.** Unknown decision / memory-state /
  failure-diagnosis labels and out-of-trace evidence IDs error in strict mode;
  incomplete memory-state coverage warns. Messages never echo hidden gold.
- [x] **Baseline reproduction commands.** `scripts/run_retrace_bench_baseline.py`
  and `scripts/run_retrace_bench_ablation.py` run against the v1.0 splits.
  Headline baselines on the v1.0 splits are **not yet regenerated** (see
  Remaining work); legacy pre-v1.0 result docs are retained for provenance only.
- [ ] **Manual validation pass (v1.0).** No human validation has been performed
  on the v1.0 splits yet. `manual_validation_protocol.md` and
  `manual_validation_sample_88.md` describe the protocol; the prior
  `manual_validation_report.md` pass was over a legacy pre-v1.0 split and does
  **not** carry over.
- [x] **Sample predictions.** `examples/retrace_bench/sample_predictions.jsonl`
  (complete `calibration_80_en` submission) + quickstart README.
- [x] **Tests.** `tests/retrace_bench/test_public_api.py`,
  `test_prediction_evaluator_cli.py`, `test_hf_package_readme.py`.
- [x] **Deterministic dataset validators.**
  `scripts/validate_retrace_bench_dataset.py` (reference integrity, hygiene,
  coverage, distribution gates).

## Remaining work

- [ ] **Human annotation for `realistic_100_en`** (fill
  `annotations_template.jsonl`; flip `annotation_status` from `pending`).
- [ ] **Full offline baseline + oracle rerun on the v1.0 splits** (`main`,
  `hard`), with results rendered into a v1.0 baseline-results doc.
- [ ] **At least one real LLM baseline** end-to-end on `main_3000_en`
  (`llm_json_answerer` needs a provider/API key).
- [ ] **Optional real memory-framework baseline** (genuine Mem0/Graphiti-style
  system rather than the in-repo heuristic).
- [ ] **Final paper tables/figures** assembled from the `main` / `hard` splits
  (per-domain / per-failure-mode breakdowns rendered, baseline plot refreshed).
- [ ] **Hugging Face upload** of the v1.0 package (not done in this pass).

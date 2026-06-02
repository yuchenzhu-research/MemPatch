# ReTrace-Bench — Artifact Checklist

Status of the benchmark artifact track. `[x]` done, `[ ]` remaining.

- [x] **Hugging Face dataset card verified.** Splits `test` (800) / `validation`
  (80) / `train` (3000) / `dev` (400) load via `datasets.load_dataset`; JSON
  string columns parse with `json.loads`; card states the strong "validation =
  viewer compatibility only, not model selection" semantics and links back to the
  repository and evaluator. Package source:
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
  and `scripts/run_retrace_bench_ablation.py`; results pre-rendered in
  `docs/retrace_bench/baseline_results_test_800_templateheldout_en.md`.
- [x] **Manual validation pass.**
  `docs/retrace_bench/manual_validation_report.md` records a completed
  project-author pass over the 88-cell stratified sample; protocol and sample
  index live in `manual_validation_protocol.md` and
  `manual_validation_sample_88.md`.
- [x] **Sample predictions.** `examples/retrace_bench/sample_predictions.jsonl`
  (complete `sample_80_hard_en` submission) + quickstart README.
- [x] **Tests.** `tests/retrace_bench/test_public_api.py`,
  `test_prediction_evaluator_cli.py`, `test_hf_package_readme.py`.
- [x] **Deterministic dataset validators.**
  `scripts/validate_retrace_bench_dataset.py` (reference integrity, hygiene,
  coverage, distribution gates).

## Remaining work

- [ ] **At least one real LLM baseline** end-to-end on
  `test_800_templateheldout_en` (`llm_json_answerer` needs a provider/API key).
- [ ] **Optional real memory-framework baseline** (genuine Mem0/Graphiti-style
  system rather than the in-repo heuristic).
- [ ] **Final paper tables/figures** assembled from the canonical split
  (per-domain / per-failure-mode breakdowns rendered, baseline plot refreshed).

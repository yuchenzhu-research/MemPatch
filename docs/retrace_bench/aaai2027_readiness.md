# ReTrace-Bench AAAI 2027 Readiness

**Track:** ReTrace-Bench is an **independent benchmark / resource paper**, not an
evaluation component of ReTrace-Learn.

**Target:** AAAI 2027 main, fallback ACL 2027 main.

## Already strong

- **Task design.** Agent memory revision reliability under evolving evidence,
  scored through four structured views (decision / memory-state / evidence /
  diagnosis).
- **Taxonomy.** 8 domains × 11 failure modes × L1–L4 difficulty, centralized in
  `benchmark/retrace_bench/general_taxonomy.py`.
- **Deterministic blueprint construction.** Hidden labels are derived from a
  blueprint, not inferred by an LLM, so they are reproducible and auditable.
- **Validators.** `scripts/validate_retrace_bench_dataset.py` enforces reference
  integrity, public-text hygiene, task coverage, and distribution gates.
- **Template-heldout split.** `data/retrace_bench/test_800_templateheldout_en/`
  is template-held out from train/dev with leakage checks.
- **Leakage checks.** Template-signature overlap and template-lookup probes
  (see `template_lookup_test_800_templateheldout_en.md`,
  `template_signature_report.md`, `split_leakage_report.md`).
- **Baseline suite.** Sanity, retrieval, and memory-architecture baselines plus
  a gold-replay oracle reference, with an explicit oracle boundary in the runner.

## Fixed today

- **Oracle decision gold replay.** `retrace_oracle_engine` now replays
  `hidden_gold.expected_decision` instead of reconstructing the decision from a
  hard-coded `failure_mode -> decision` mapping. On
  `test_800_templateheldout_en`, oracle decision accuracy went from `0.294` to
  `1.000` and decision macro-F1 from `0.232` to `1.000` (non-answer accuracy
  `0.132 -> 1.000`). See `baseline_results_test_800_templateheldout_en.md`.
- **Headline metric constants.** Added `HEADLINE_METRICS` and
  `AUXILIARY_METRICS` to `benchmark/retrace_bench/scorers_general.py`;
  `decision_macro_f1` is the primary decision metric, and
  `black_box_decision_accuracy` is explicitly auxiliary (it can be dominated by
  the majority `use_current_memory` class). `aggregate_metrics` now also returns
  grouped `headline_metrics` / `auxiliary_metrics` / `all_metrics` views
  (backward compatible: the flat `metrics` key is preserved).
- **Centralized decision label space.** `DECISIONS` and `NON_ANSWER_DECISIONS`
  live in `general_taxonomy.py`; scorers and the template-heldout generator
  import them rather than redefining their own copies (`ALL_DECISIONS` is kept
  as a backward-compatible alias).
- **Split-role clarification.** `test_800_templateheldout_en` (canonical
  paper-facing test), `test_800_en` (prototype/diagnostic only),
  `sample_80_hard_en` (calibration/quickstart; HF `validation` is viewer
  compatibility only, not model selection), and `train_3000_en`/`dev_400_en`
  (supervision pools, not test sets) are described consistently across
  `benchmark_paper.md`, `dataset_design.md`, `data/README.md`, and the HF README.
- **Safer related-work framing.** `benchmark_paper.md` now frames ReTrace-Bench
  as complementary to LongMemEval, MemBench, LoCoMo (and optionally
  MemoryAgentBench / EvoMemBench) and drops "none of these directly evaluate..."
  in favor of "isolates", "complementary", and "operational evaluation target".

## Remains after today

- **At least one real LLM baseline** run end-to-end (`llm_json_answerer`
  requires a provider/API key) on `test_800_templateheldout_en`, reported in the
  baseline table.
- **Optional real memory-framework baseline** (e.g. a genuine Mem0/Graphiti-style
  system rather than the in-repo `mem0_style` heuristic).
- **Completed manual validation report.** Fill in
  `manual_validation_sample_88.md` per `manual_validation_protocol.md` with a
  human reviewer pass.
- **Final paper tables.** Camera-ready headline + auxiliary tables and per-domain
  / per-failure-mode breakdowns assembled from the canonical split.

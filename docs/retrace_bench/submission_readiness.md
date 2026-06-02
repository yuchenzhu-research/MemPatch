# ReTrace-Bench Submission Readiness

**Track:** ReTrace-Bench is an **independent benchmark / resource paper**, not an
evaluation component of ReTrace-Learn. This document is an internal readiness
checklist for the submission-ready benchmark artifact; it intentionally states
no target venue (the benchmark paper is prepared for blind review).

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

## Benchmark artifact interface

ReTrace-Bench now ships a professional, paper-ready evaluation interface (not
just a data dump):

- **HF dataset card/package checked.** The published dataset card was inspected
  via `huggingface_hub` and `datasets.load_dataset`: splits `test` (800) /
  `validation` (80) / `train` (3000) / `dev` (400) load, map to the expected
  on-disk paths, and the JSON string columns parse with `json.loads`. The
  repo-side packaging template (`scripts/package_hf_retrace_bench.py`), the
  checked-in `release/huggingface/ReTrace-Bench/README.md`, and the live card
  all state the strong "viewer compatibility only — not model/checkpoint
  selection" semantics.
- **Strict validation contract.** In `strict=True`, `evaluate_predictions`
  rejects unknown `decision` labels, unknown `memory_state` statuses, evidence
  IDs absent from `public_input.event_trace`, and `failure_diagnosis` values
  that are neither a canonical `FAILURE_MODES` label nor a documented alias.
  Incomplete `memory_state` coverage of the visible `initial_memory` IDs is a
  warning (never an error), so partial submissions are still scored. Validation
  messages reference only model-visible IDs and never echo hidden gold.
- **Official prediction schema.** Documented in `examples/retrace_bench/` and the
  HF card: `decision` (5 labels), `memory_state` (`memory_id -> status`, 8
  labels), `evidence_event_ids` (from `public_input.event_trace`),
  `failure_diagnosis` (11 labels), and free-text `answer`; canonical nested and
  flat forms both accepted.
- **Official Python API.** `benchmark/retrace_bench/api.py` exposes
  `load_scenarios`, `load_predictions`, `normalize_prediction`,
  `evaluate_predictions(strict=...)`, and re-exports `HEADLINE_METRICS`,
  `AUXILIARY_METRICS`, `DECISIONS`, `MEMORY_STATUSES`, `FAILURE_MODES`. It wraps
  the existing scorer (`score_prediction` / `aggregate_metrics`) without changing
  scoring behavior.
- **Official evaluator CLI.** `scripts/evaluate_retrace_bench_predictions.py`
  scores an external predictions file against a split, requires no API keys, runs
  no model, and supports `--strict/--no-strict`, `--allow-missing`,
  `--out-metrics`, `--out-scored`, and `--print-table`.
- **Example predictions.** `examples/retrace_bench/sample_predictions.jsonl` (a
  complete calibration-split submission) plus a README quickstart.
- **Canonical metric constants in tooling.**
  `scripts/run_retrace_bench_ablation.py` now imports `HEADLINE_METRICS` /
  `AUXILIARY_METRICS`; `decision_macro_f1` leads the table and
  `black_box_decision_accuracy` is emitted only as `decision_acc_aux`.
- **Tests.** `tests/retrace_bench/test_public_api.py`,
  `test_prediction_evaluator_cli.py`, and `test_hf_package_readme.py` cover the
  API, CLI, strict/non-strict validation, and the HF template wording.
- **Packaging note.** `benchmark.retrace_bench` is imported via `PYTHONPATH=.`
  (it is intentionally not added to the `retracemem` setuptools discovery to
  avoid perturbing the ReTrace-Learn install); all benchmark commands are
  documented with the `PYTHONPATH=.` prefix.

## Blind-review safety

- Benchmark-facing docs (`benchmark/README.md`, `docs/retrace_bench/*`, the HF
  card) state no target venue and no conference strategy; `grep` for
  `AAAI|ACL 20|ICLR|ICML|NeurIPS|fallback|target:` returns only unrelated
  technical uses (e.g. a regex "fallback" comment, "fallback for unseen
  template signatures").
- The blinded paper workspace lives in `papers/retrace_bench/` and uses
  `[anonymized repository]` / `[anonymized dataset link]` placeholders, no
  author names, and no public usernames; see
  `papers/retrace_bench/blind_review_checklist.md`.
- Real artifact links (GitHub repo, HF dataset) remain only in the public
  artifact docs (README / HF card), which are de-anonymizable distribution
  surfaces, not the manuscript text.

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

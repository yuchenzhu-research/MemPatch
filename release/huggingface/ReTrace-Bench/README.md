---
license: cc-by-4.0
language:
- en
pretty_name: ReTrace-Bench
task_categories:
- question-answering
- text-classification
- text-generation
tags:
- agent-memory
- llm-agents
- benchmark
- memory-revision
- long-term-memory
- reliability
---

# ReTrace-Bench

ReTrace-Bench is a synthetic English benchmark for evaluating agent memory
revision reliability. It tests whether an agent can use event evidence to keep,
update, block, restore, forget, or reject long-term memory facts without reusing
stale, out-of-scope, unsupported, or policy-invalid memory.

## Dataset Splits

The paper-facing held-out benchmark split is:

- `data/test_800_templateheldout_en/`

Use this split for final benchmark evaluation only. Models and prompts should
not train, prompt-tune, select checkpoints, optimize policies, or perform model
selection on `test_800_templateheldout_en`.

The previous repository split `test_800_en` is prototype/diagnostic only and is
not included as the final benchmark split in this release package.

This release also includes:

- `data/sample_80_hard_en/` - a small hard sample for smoke tests and examples.

The upstream ReTrace repository also contains `train_3000_en` and `dev_400_en`
as synthetic supervision/selection pools. They are not included in this
Hugging Face benchmark release and must not be treated as held-out benchmark
tests.

## Main Task Views

Each scenario contains visible workflow evidence and four task views:

- `black_box_task`
- `memory_state_task`
- `evidence_retrieval_task`
- `diagnostic_task`

The hidden labels are for scoring and audit only. They should not be exposed to
systems under evaluation.

## Primary Metrics

The intended headline metrics are:

- `decision_macro_f1`
- `non_answer_decision_accuracy`
- `memory_state_accuracy`
- `evidence_f1`
- `failure_diagnosis_accuracy`
- `stale_reuse_rate`

Diagnostic documents and scoring helpers are included under `docs/` and
`schema/`.

## Template Shortcut Diagnostic

The template-lookup diagnostic is a shortcut-leakage probe, not a deployable
memory baseline. On `test_800_templateheldout_en`, the train-signature lookup
has:

- coverage: `0.000`
- decision accuracy: `0.291`
- decision macro-F1: `0.090`

The candidate held-out split has zero train-to-test scenario-signature overlap
and zero train-to-test event-template overlap in the included diagnostic report.

## Baselines

Offline baseline results are included in:

- `docs/baseline_results_test_800_templateheldout_en.md`

Oracle state/evidence/diagnosis rows are diagnostic upper-bound paths, not
deployable or comparable baselines. In the current offline results, oracle
memory-state, evidence, and diagnosis scores are high while oracle black-box
decision accuracy is low. This is expected for this diagnostic row because the
oracle path primarily replays typed state/evidence/diagnosis structure through
the deterministic authorization path, while the benchmark's black-box decision
labels are intentionally decorrelated from primary failure mode shortcuts.

## Schema

Important schema and scoring references:

- `schema/general_schema.py`
- `schema/general_taxonomy.py`
- `schema/scorers_general.py`
- `docs/schema_v2_proposal.md`
- `docs/dataset_design.md`
- `docs/failure_modes.md`
- `docs/metrics_v2.md`
- `docs/quality_gates_v2.md`

## License

This dataset release is licensed under Creative Commons Attribution 4.0
International (CC BY 4.0).

See `LICENSE` for the license pointer.

## Source

Source repository: `yuchenzhu-research/ReTrace`

Source commit: `7e11bf67097f31bf2c072236331b2229d662a609`

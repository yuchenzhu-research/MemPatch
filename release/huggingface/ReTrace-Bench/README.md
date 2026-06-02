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
configs:
- config_name: default
  data_files:
  - split: test
    path: test.jsonl
---

# ReTrace-Bench

ReTrace-Bench is a synthetic English benchmark for evaluating agent memory
revision reliability. It tests whether an agent can use event evidence to keep,
update, block, restore, forget, or reject long-term memory facts without reusing
stale, out-of-scope, unsupported, or policy-invalid memory.

## Dataset Splits

This Hugging Face release contains only the paper-facing held-out benchmark
test split:

- `test.jsonl`

Use this split for final benchmark evaluation only. Models and prompts should not
train, prompt-tune, select checkpoints, optimize policies, or perform model
selection on this test split.

The previous repository split `test_800_en` is prototype/diagnostic only and is
not included as the final benchmark split in this release package.

The upstream ReTrace repository also contains `train_3000_en` and `dev_400_en`
as synthetic supervision/selection pools. They are not included in this Hugging
Face benchmark release and must not be treated as held-out benchmark tests.

## Viewer-Friendly Format

The upstream benchmark records contain nested event traces, memory states, and
rubrics. Hugging Face Dataset Viewer expects stable Arrow columns, so this
release stores complex nested objects as JSON strings:

- `event_trace_json`
- `initial_memory_json`
- `tasks_json`
- `expected_memory_state_json`
- `rubric_json`
- `metadata_json`

This preserves the full test scenario while keeping the dataset browser
scrollable and queryable. The stable scalar/list columns include scenario IDs,
domain, failure mode, expected decision, expected answer, evidence IDs, and
diagnosis.

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

Diagnostic documents and scoring helpers live in the upstream ReTrace
repository.

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

Oracle state/evidence/diagnosis rows are diagnostic upper-bound paths, not
deployable or comparable baselines. In the current offline results, oracle
memory-state, evidence, and diagnosis scores are high while oracle black-box
decision accuracy is low. This is expected for this diagnostic row because the
oracle path primarily replays typed state/evidence/diagnosis structure through
the deterministic authorization path, while the benchmark's black-box decision
labels are intentionally decorrelated from primary failure mode shortcuts.

## Schema

Important schema, scoring, leakage, and baseline references are available in
the upstream ReTrace repository under `benchmark/retrace_bench/`,
`docs/retrace_bench/`, and `release/huggingface/ReTrace-Bench/`.

## License

This dataset release is licensed under Creative Commons Attribution 4.0
International (CC BY 4.0).

See `LICENSE` for the license pointer.

## Source

Source repository: `yuchenzhu-research/ReTrace`

Source data commit: `7e11bf67097f31bf2c072236331b2229d662a609`

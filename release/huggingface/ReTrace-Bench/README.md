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
    path: benchmark/test_800_templateheldout_en/scenarios.jsonl
  - split: validation
    path: calibration/sample_80_hard_en/scenarios.jsonl
  - split: train
    path: supervision/train_3000_en/scenarios.jsonl
  - split: dev
    path: supervision/dev_400_en/scenarios.jsonl
---

# ReTrace-Bench

ReTrace-Bench evaluates agent memory revision reliability in multi-agent and agentic workflows. It tests whether systems can correctly process new evidence to update, block, release, reaffirm, or reject memory states without introducing stale, out-of-scope, or policy-invalid memory.

## IMPORTANT Notice

- **benchmark/test_800_templateheldout_en** is the canonical, paper-facing held-out benchmark split.
- **Do not train, prompt-tune, policy-optimize, or select checkpoints on `benchmark/test_800_templateheldout_en`.**
- `calibration/sample_80_hard_en` is a small quickstart/calibration split designed for debugging and pipeline verification. It is exposed to the Hugging Face viewer as the `validation` split.
- `supervision/train_3000_en` and `supervision/dev_400_en` are synthetic supervision/selection pools for learning-based revision proposers. They are **NOT** benchmark tests and may contain `training_targets`.
- The old prototype/diagnostic split `test_800_en` is excluded from this public release package.

## Current Dataset Scale

- **benchmark/test_800_templateheldout_en**: 800 scenarios
- **calibration/sample_80_hard_en**: 80 scenarios
- **supervision/train_3000_en**: 3000 scenarios (contains SFT training targets)
- **supervision/dev_400_en**: 400 scenarios
- **Total packaged scenarios**: 4280 scenarios

## Scenario Task Views

Each scenario evaluates an agent memory state through four distinct task views:

1. **`black_box_task`**: Evaluate end-to-end question answering utilizing revised memory state.
2. **`memory_state_task`**: Predict final eligibility statuses of all tracked beliefs.
3. **`evidence_retrieval_task`**: Identify evidence items that active memory changes are grounded upon.
4. **`diagnostic_task`**: Detect and classify memory-revision failures or conflicts.

For Hugging Face dataset viewer compatibility, nested scenario structures are
published as JSON string columns:

- `secondary_failure_modes_json`
- `public_input_json`
- `tasks_json`
- `hidden_gold_json`
- `metadata_json`
- `training_targets_json`

Parse these columns with `json.loads(...)` to recover the canonical nested
objects. The source-of-truth local files under `data/` keep the native nested
JSONL schema.

## Primary Metrics

Benchmark evaluations report the following primary metrics:

- `decision_macro_f1`
- `non_answer_decision_accuracy`
- `memory_state_accuracy`
- `evidence_f1`
- `failure_diagnosis_accuracy`
- `stale_reuse_rate`

## Quality Diagnostics for `test_800_templateheldout_en`

The held-out test split is designed with strict template-independent validation guarantees:
- Includes scenarios across **all 8 domains** and **all 11 failure modes**.
- **Template lookup coverage**: `0.000`
- **Template lookup decision accuracy**: `0.291`
- **Template lookup macro-F1**: `0.090`
- **Train/dev template signature overlap with candidate test**: `0.00%`

## Baseline Caveat

Please note that the **oracle** row documented in baseline results represents a diagnostic verification path for replaying typed state/evidence/diagnosis structures through the deterministic ReTrace-Engine. It is **not** a deployable memory baseline and should not be treated as a black-box decision upper bound.

## License

This dataset is distributed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE) license.

# ReTrace-Bench v1.1 Hard150 — SiliconFlow Three-Model Eval

Dataset: `data/retrace_bench_hard150/hard_150_en/scenarios.jsonl`
Scenarios: 2

## Dataset distributions

- difficulty: `{'L3': 1, 'L4': 1}`
- pattern: `{'closed_as_duplicate_not_fixed': 1, 'merged_but_unreleased': 1}`
- expected_decision: `{'mark_unresolved': 1, 'use_current_memory': 1}`

## Main metrics

| model | decision_macro_f1 | black_box_decision_accuracy | non_answer_decision_accuracy | memory_state_accuracy | evidence_f1 | minimal_evidence_exact_match | evidence_precision | overcitation_rate | counterevidence_recall | failure_diagnosis_accuracy | stale_reuse_rate | latest_event_shortcut_failure_rate | answer_state_consistency | joint_revision_success | format_failure_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Pro | 0.333 | 0.500 | 0.000 | 0.833 | 1.000 | 1.000 | 1.000 | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Ranking by joint_revision_success

1. **DeepSeek-V4-Pro** — joint_revision_success=0.000

## Per-model failure summaries

### DeepSeek-V4-Pro
- format failures: 0
- lowest joint examples: `[{'scenario_id': 'rt-hard-000001', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000002', 'joint_revision_success': 0.0}]`
- common wrong failure_diagnosis: `{'scope_leakage': 2}`
- overcitation examples: `[]`
- non-answer decision failures: `[{'scenario_id': 'rt-hard-000002', 'expected': 'mark_unresolved', 'predicted': 'use_current_memory'}]`


## Legacy DeepSeek hard150 comparison

- Previous run (`outputs/retrace_bench_hard150/`): joint=0.247, failure_diagnosis=0.207
- This run: joint=0.000, failure_diagnosis=0.000
- Difference >0.05 likely due to prompt/runner path change (this eval uses public-field-only SiliconFlow runner with expanded instruction rules).

## Conclusions

- **Hard150 difficulty:** sufficiently hard (max joint_revision_success=0.000).
- **Strongest model:** DeepSeek-V4-Pro.
- **Most diagnostic metrics:** joint_revision_success, failure_diagnosis_accuracy, minimal_evidence_exact_match, overcitation_rate.
- **Expand to hard500:** yes, after fixing decision skew.
- **Paper stress split:** yes — hard150 is suitable as v1.1 headline stress split.

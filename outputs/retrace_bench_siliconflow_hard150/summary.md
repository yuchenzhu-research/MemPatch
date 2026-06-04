# ReTrace-Bench v1.1 Hard150 — SiliconFlow Three-Model Eval

Dataset: `data/retrace_bench_hard150/hard_150_en/scenarios.jsonl`
Scenarios: 150

## Dataset distributions

- difficulty: `{'L3': 75, 'L4': 75}`
- pattern: `{'authority_conflict': 10, 'backport_only_fix': 10, 'branch_scope_leakage': 10, 'ci_failed_after_claim': 10, 'closed_as_duplicate_not_fixed': 10, 'docs_ahead_of_code': 10, 'label_state_mismatch': 10, 'maintainer_correction_over_user_claim': 10, 'merged_but_unreleased': 10, 'multi_memory_coupling': 10, 'negative_evidence_required': 10, 'release_then_revert': 10, 'security_policy_override': 10, 'stale_comment_after_new_release': 10, 'version_scope_leakage': 10}`
- expected_decision: `{'mark_unresolved': 25, 'refuse_due_to_policy': 10, 'use_current_memory': 115}`

## Main metrics

| model | decision_macro_f1 | black_box_decision_accuracy | non_answer_decision_accuracy | memory_state_accuracy | evidence_f1 | minimal_evidence_exact_match | evidence_precision | overcitation_rate | counterevidence_recall | failure_diagnosis_accuracy | stale_reuse_rate | latest_event_shortcut_failure_rate | answer_state_consistency | joint_revision_success | format_failure_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kimi-K2.6 | 0.288 | 0.760 | 0.000 | 0.853 | 0.661 | 0.320 | 0.566 | 0.434 | 0.667 | 0.267 | 0.040 | 0.033 | 0.567 | 0.133 | 0.000 |
| GLM-5.1 | 0.665 | 0.640 | 0.600 | 0.849 | 0.802 | 0.407 | 0.806 | 0.194 | 0.640 | 0.400 | 0.013 | 0.000 | 0.407 | 0.233 | 0.000 |
| DeepSeek-V4-Pro | 0.285 | 0.747 | 0.000 | 0.851 | 0.809 | 0.493 | 0.737 | 0.263 | 0.687 | 0.167 | 0.000 | 0.027 | 0.500 | 0.273 | 0.000 |

## Ranking by joint_revision_success

1. **DeepSeek-V4-Pro** — joint_revision_success=0.273
2. **GLM-5.1** — joint_revision_success=0.233
3. **Kimi-K2.6** — joint_revision_success=0.133

## Per-model failure summaries

### Kimi-K2.6
- format failures: 0
- lowest joint examples: `[{'scenario_id': 'rt-hard-000001', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000002', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000004', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000005', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000006', 'joint_revision_success': 0.0}]`
- common wrong failure_diagnosis: `{'scope_leakage': 58, 'stale_memory_reuse': 40, 'wrong_source_attribution': 6, 'under_update': 6}`
- overcitation examples: `[{'scenario_id': 'rt-hard-000006', 'overcitation_rate': 0.75}, {'scenario_id': 'rt-hard-000010', 'overcitation_rate': 0.6}, {'scenario_id': 'rt-hard-000011', 'overcitation_rate': 0.75}, {'scenario_id': 'rt-hard-000012', 'overcitation_rate': 0.8333333333333334}, {'scenario_id': 'rt-hard-000013', 'overcitation_rate': 0.75}]`
- non-answer decision failures: `[{'scenario_id': 'rt-hard-000002', 'expected': 'mark_unresolved', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000007', 'expected': 'mark_unresolved', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000008', 'expected': 'mark_unresolved', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000009', 'expected': 'refuse_due_to_policy', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000017', 'expected': 'mark_unresolved', 'predicted': 'use_current_memory'}]`

### GLM-5.1
- format failures: 0
- lowest joint examples: `[{'scenario_id': 'rt-hard-000001', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000002', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000003', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000004', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000007', 'joint_revision_success': 0.0}]`
- common wrong failure_diagnosis: `{'scope_leakage': 49, 'stale_memory_reuse': 25, 'wrong_source_attribution': 6, 'conflict_collapse': 5, 'over_update': 3}`
- overcitation examples: `[{'scenario_id': 'rt-hard-000012', 'overcitation_rate': 0.75}, {'scenario_id': 'rt-hard-000014', 'overcitation_rate': 0.75}, {'scenario_id': 'rt-hard-000029', 'overcitation_rate': 0.6666666666666666}, {'scenario_id': 'rt-hard-000044', 'overcitation_rate': 0.75}, {'scenario_id': 'rt-hard-000057', 'overcitation_rate': 0.6666666666666666}]`
- non-answer decision failures: `[{'scenario_id': 'rt-hard-000007', 'expected': 'mark_unresolved', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000008', 'expected': 'mark_unresolved', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000023', 'expected': 'mark_unresolved', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000037', 'expected': 'mark_unresolved', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000038', 'expected': 'mark_unresolved', 'predicted': 'use_current_memory'}]`

### DeepSeek-V4-Pro
- format failures: 0
- lowest joint examples: `[{'scenario_id': 'rt-hard-000002', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000004', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000006', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000007', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000008', 'joint_revision_success': 0.0}]`
- common wrong failure_diagnosis: `{'scope_leakage': 114, 'stale_memory_reuse': 10, 'conflict_collapse': 1}`
- overcitation examples: `[{'scenario_id': 'rt-hard-000008', 'overcitation_rate': 0.6}, {'scenario_id': 'rt-hard-000009', 'overcitation_rate': 0.6666666666666666}, {'scenario_id': 'rt-hard-000011', 'overcitation_rate': 0.75}, {'scenario_id': 'rt-hard-000024', 'overcitation_rate': 0.8}, {'scenario_id': 'rt-hard-000029', 'overcitation_rate': 0.75}]`
- non-answer decision failures: `[{'scenario_id': 'rt-hard-000002', 'expected': 'mark_unresolved', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000007', 'expected': 'mark_unresolved', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000008', 'expected': 'mark_unresolved', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000009', 'expected': 'refuse_due_to_policy', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000017', 'expected': 'mark_unresolved', 'predicted': 'use_current_memory'}]`


## Legacy DeepSeek hard150 comparison

- Previous run (`outputs/retrace_bench_hard150/`): joint=0.247, failure_diagnosis=0.207
- This run: joint=0.273, failure_diagnosis=0.167
- Results are broadly consistent with the prior hard150 DeepSeek run.

## Conclusions

- **Hard150 difficulty:** sufficiently hard (max joint_revision_success=0.273).
- **Strongest model:** DeepSeek-V4-Pro.
- **Most diagnostic metrics:** joint_revision_success, failure_diagnosis_accuracy, minimal_evidence_exact_match, overcitation_rate.
- **Expand to hard500:** yes, after fixing decision skew.
- **Paper stress split:** yes — hard150 is suitable as v1.1 headline stress split.

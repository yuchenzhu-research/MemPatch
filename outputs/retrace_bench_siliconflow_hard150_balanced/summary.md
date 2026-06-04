# ReTrace-Bench v1.1 Hard150 — SiliconFlow Three-Model Eval

Dataset: `data/retrace_bench_hard150_balanced/hard_150_en/scenarios.jsonl`
Scenarios: 150

## Dataset distributions

- difficulty: `{'L3': 75, 'L4': 75}`
- pattern: `{'authority_conflict': 5, 'backport_only_fix': 7, 'branch_scope_leakage': 8, 'ci_failed_after_claim': 40, 'closed_as_duplicate_not_fixed': 10, 'docs_ahead_of_code': 8, 'label_state_mismatch': 7, 'maintainer_correction_over_user_claim': 7, 'merged_but_unreleased': 8, 'multi_memory_coupling': 7, 'negative_evidence_required': 5, 'release_then_revert': 8, 'security_policy_override': 15, 'stale_comment_after_new_release': 7, 'version_scope_leakage': 8}`
- expected_decision: `{'ask_clarification': 18, 'escalate': 12, 'mark_unresolved': 30, 'refuse_due_to_policy': 15, 'use_current_memory': 75}`

## Main metrics

| model | decision_macro_f1 | black_box_decision_accuracy | non_answer_decision_accuracy | memory_state_accuracy | evidence_f1 | minimal_evidence_exact_match | evidence_precision | overcitation_rate | counterevidence_recall | failure_diagnosis_accuracy | stale_reuse_rate | latest_event_shortcut_failure_rate | answer_state_consistency | joint_revision_success | format_failure_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kimi-K2.6 | 0.133 | 0.500 | 0.000 | 0.787 | 0.694 | 0.387 | 0.618 | 0.382 | 0.667 | 0.180 | 0.020 | 0.047 | 0.360 | 0.133 | 0.000 |
| GLM-5.1 | 0.342 | 0.533 | 0.373 | 0.791 | 0.837 | 0.587 | 0.823 | 0.177 | 0.713 | 0.233 | 0.007 | 0.000 | 0.307 | 0.193 | 0.000 |
| DeepSeek-V4-Pro | 0.195 | 0.513 | 0.040 | 0.789 | 0.823 | 0.520 | 0.748 | 0.252 | 0.780 | 0.153 | 0.000 | 0.020 | 0.320 | 0.227 | 0.000 |

## Ranking by joint_revision_success

1. **DeepSeek-V4-Pro** — joint_revision_success=0.227
2. **GLM-5.1** — joint_revision_success=0.193
3. **Kimi-K2.6** — joint_revision_success=0.133

## Per-model failure summaries

### Kimi-K2.6
- format failures: 0
- lowest joint examples: `[{'scenario_id': 'rt-hard-000001', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000002', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000003', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000004', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000005', 'joint_revision_success': 0.0}]`
- common wrong failure_diagnosis: `{'scope_leakage': 84, 'stale_memory_reuse': 34, 'under_update': 5}`
- overcitation examples: `[{'scenario_id': 'rt-hard-000002', 'overcitation_rate': 0.5714285714285714}, {'scenario_id': 'rt-hard-000004', 'overcitation_rate': 1.0}, {'scenario_id': 'rt-hard-000005', 'overcitation_rate': 0.75}, {'scenario_id': 'rt-hard-000011', 'overcitation_rate': 0.75}, {'scenario_id': 'rt-hard-000016', 'overcitation_rate': 0.75}]`
- non-answer decision failures: `[{'scenario_id': 'rt-hard-000004', 'expected': 'refuse_due_to_policy', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000006', 'expected': 'mark_unresolved', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000007', 'expected': 'ask_clarification', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000008', 'expected': 'escalate', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000013', 'expected': 'mark_unresolved', 'predicted': 'use_current_memory'}]`

### GLM-5.1
- format failures: 0
- lowest joint examples: `[{'scenario_id': 'rt-hard-000002', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000003', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000004', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000005', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000006', 'joint_revision_success': 0.0}]`
- common wrong failure_diagnosis: `{'scope_leakage': 57, 'stale_memory_reuse': 50, 'wrong_source_attribution': 6, 'over_update': 1, 'conflict_collapse': 1}`
- overcitation examples: `[{'scenario_id': 'rt-hard-000004', 'overcitation_rate': 0.8}, {'scenario_id': 'rt-hard-000009', 'overcitation_rate': 0.6666666666666666}, {'scenario_id': 'rt-hard-000011', 'overcitation_rate': 0.75}, {'scenario_id': 'rt-hard-000016', 'overcitation_rate': 0.8}, {'scenario_id': 'rt-hard-000018', 'overcitation_rate': 1.0}]`
- non-answer decision failures: `[{'scenario_id': 'rt-hard-000004', 'expected': 'refuse_due_to_policy', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000006', 'expected': 'mark_unresolved', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000007', 'expected': 'ask_clarification', 'predicted': 'mark_unresolved'}, {'scenario_id': 'rt-hard-000008', 'expected': 'escalate', 'predicted': 'mark_unresolved'}, {'scenario_id': 'rt-hard-000014', 'expected': 'escalate', 'predicted': 'mark_unresolved'}]`

### DeepSeek-V4-Pro
- format failures: 0
- lowest joint examples: `[{'scenario_id': 'rt-hard-000002', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000003', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000004', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000005', 'joint_revision_success': 0.0}, {'scenario_id': 'rt-hard-000006', 'joint_revision_success': 0.0}]`
- common wrong failure_diagnosis: `{'scope_leakage': 110, 'stale_memory_reuse': 15, 'conflict_collapse': 2}`
- overcitation examples: `[{'scenario_id': 'rt-hard-000004', 'overcitation_rate': 0.8}, {'scenario_id': 'rt-hard-000008', 'overcitation_rate': 0.6666666666666666}, {'scenario_id': 'rt-hard-000016', 'overcitation_rate': 0.8}, {'scenario_id': 'rt-hard-000018', 'overcitation_rate': 0.6666666666666666}, {'scenario_id': 'rt-hard-000034', 'overcitation_rate': 0.6666666666666666}]`
- non-answer decision failures: `[{'scenario_id': 'rt-hard-000004', 'expected': 'refuse_due_to_policy', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000006', 'expected': 'mark_unresolved', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000007', 'expected': 'ask_clarification', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000008', 'expected': 'escalate', 'predicted': 'use_current_memory'}, {'scenario_id': 'rt-hard-000013', 'expected': 'mark_unresolved', 'predicted': 'use_current_memory'}]`


## Legacy DeepSeek hard150 comparison

- Previous run (`outputs/retrace_bench_hard150/`): joint=0.273, failure_diagnosis=0.167
- This run: joint=0.227, failure_diagnosis=0.153
- Results are broadly consistent with the prior hard150 DeepSeek run.

## Conclusions

- **Hard150 difficulty:** sufficiently hard (max joint_revision_success=0.227).
- **Strongest model:** DeepSeek-V4-Pro.
- **Most diagnostic metrics:** joint_revision_success, failure_diagnosis_accuracy, minimal_evidence_exact_match, overcitation_rate.
- **Expand to hard500:** yes, after fixing decision skew.
- **Paper stress split:** yes — hard150 is suitable as v1.1 headline stress split.

# ReTrace-Bench Hard150 Scale Test Summary

**Status:** scale test only — not a final release.

Dataset: `data/retrace_bench_hard150/hard_150_en/scenarios.jsonl`
Scenarios: 150

## Distributions

### Pattern
```json
{
  "authority_conflict": 10,
  "backport_only_fix": 10,
  "branch_scope_leakage": 10,
  "ci_failed_after_claim": 10,
  "closed_as_duplicate_not_fixed": 10,
  "docs_ahead_of_code": 10,
  "label_state_mismatch": 10,
  "maintainer_correction_over_user_claim": 10,
  "merged_but_unreleased": 10,
  "multi_memory_coupling": 10,
  "negative_evidence_required": 10,
  "release_then_revert": 10,
  "security_policy_override": 10,
  "stale_comment_after_new_release": 10,
  "version_scope_leakage": 10
}
```

### Failure mode
```json
{
  "conflict_collapse": 20,
  "failure_to_release_or_restore": 5,
  "over_update": 15,
  "policy_violation": 10,
  "scope_leakage": 25,
  "stale_memory_reuse": 40,
  "under_update": 15,
  "wrong_source_attribution": 20
}
```

### Expected decision
```json
{
  "mark_unresolved": 25,
  "refuse_due_to_policy": 10,
  "use_current_memory": 115
}
```

### Difficulty / source
- difficulty: `{'L3': 75, 'L4': 75}`
- source_type: `{'controlled_synthetic': 150}`
- pattern balance spread: `0`
- top decision share: `use_current_memory` @ 76.7%

## Semantic / quality notes

- Pattern counts are approximately balanced (10-10 per pattern across 15 patterns).
- Expected decision distribution is moderately skewed toward `use_current_memory` (76.7%).
- Background filler is present but moderate (avg 1.6 bg events; 7.0 total events on average).
- Failure-mode labels are pattern-bound via PATTERN_SPEC; validator + gold oracle passing indicates semantic alignment at generation time.

## Metrics

| method | decision_macro_f1 | black_box_decision_accuracy | non_answer_decision_accuracy | memory_state_accuracy | evidence_f1 | minimal_evidence_exact_match | evidence_precision | overcitation_rate | counterevidence_recall | failure_diagnosis_accuracy | stale_reuse_rate | latest_event_shortcut_failure_rate | answer_state_consistency | joint_revision_success | format_failure_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| latest_only | 0.289 | 0.767 | 0.000 | 0.522 | 0.000 | 0.000 | 0.000 | 1.000 | 0.333 | 0.100 | 0.000 | 0.233 | 0.000 | 0.000 | 0.000 |
| retrieve_all | 0.289 | 0.767 | 0.000 | 0.522 | 0.437 | 0.000 | 0.291 | 0.709 | 0.933 | 0.100 | 0.067 | 0.233 | 0.000 | 0.000 | 0.000 |
| DeepSeek-V4-Pro | 0.352 | 0.773 | 0.029 | 0.816 | 0.863 | 0.613 | 0.812 | 0.188 | 0.673 | 0.207 | 0.027 | 0.000 | 0.487 | 0.247 | 0.000 |

## Scale to hard_500

**worth_scaling_to_hard_500**

- latest_only remains weak on joint_revision_success
- retrieve_all still overcites, preserving difficulty signal
- DeepSeek joint_revision_success is low but non-zero (discriminative)
- failure_diagnosis_accuracy remains meaningful but not trivial

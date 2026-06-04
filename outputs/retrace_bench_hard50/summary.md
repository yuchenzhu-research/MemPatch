# ReTrace-Bench Hard50 Summary

Dataset: `data/retrace_bench_hard50/hard_50_en/scenarios.jsonl`
Scenarios: 50

## Metrics

| method | decision_macro_f1 | black_box_decision_accuracy | non_answer_decision_accuracy | memory_state_accuracy | evidence_f1 | minimal_evidence_exact_match | evidence_precision | overcitation_rate | counterevidence_recall | failure_diagnosis_accuracy | stale_reuse_rate | latest_event_shortcut_failure_rate | answer_state_consistency | joint_revision_success | format_failure_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| latest_only | 0.288 | 0.760 | 0.000 | 0.513 | 0.000 | 0.000 | 0.000 | 1.000 | 0.360 | 0.100 | 0.000 | 0.240 | 0.000 | 0.000 | 0.000 |
| retrieve_all | 0.288 | 0.760 | 0.000 | 0.513 | 0.453 | 0.000 | 0.305 | 0.695 | 0.940 | 0.100 | 0.060 | 0.240 | 0.000 | 0.000 | 0.000 |
| DeepSeek-V4-Pro | 0.453 | 0.760 | 0.083 | 0.807 | 0.878 | 0.620 | 0.837 | 0.163 | 0.660 | 0.200 | 0.000 | 0.000 | 0.480 | 0.260 | 0.000 |

## Recommendation

**continue_final_hardening**

- latest_only is weak on joint_revision_success as expected
- retrieve_all shows high overcitation_rate
- strong API models still struggle: DeepSeek-V4-Pro
- failure_diagnosis_accuracy is not trivially perfect

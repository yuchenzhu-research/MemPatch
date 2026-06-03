# ReTrace-Bench Hard50 Summary

Dataset: `data/retrace_bench_hard50/hard_50_en/scenarios.jsonl`
Scenarios: 50

## Metrics

| method | decision_macro_f1 | black_box_decision_accuracy | non_answer_decision_accuracy | memory_state_accuracy | evidence_f1 | minimal_evidence_exact_match | evidence_precision | overcitation_rate | counterevidence_recall | failure_diagnosis_accuracy | stale_reuse_rate | latest_event_shortcut_failure_rate | answer_state_consistency | joint_revision_success | format_failure_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| latest_only | 0.213 | 0.740 | 0.000 | 0.513 | 0.000 | 0.000 | 0.000 | 1.000 | 0.360 | 0.100 | 0.000 | 0.260 | 0.000 | 0.000 | 0.000 |
| retrieve_all | 0.213 | 0.740 | 0.000 | 0.513 | 0.453 | 0.000 | 0.305 | 0.695 | 0.940 | 0.100 | 0.000 | 0.260 | 0.000 | 0.000 | 0.000 |
| Kimi-K2.6 | 0.418 | 0.780 | 0.154 | 0.847 | 0.505 | 0.040 | 0.384 | 0.616 | 0.680 | 0.140 | 0.000 | 0.000 | 0.600 | 0.040 | 0.000 |
| GLM-5.1 | 0.464 | 0.720 | 0.231 | 0.767 | 0.791 | 0.460 | 0.768 | 0.192 | 0.680 | 0.100 | 0.000 | 0.000 | 0.380 | 0.180 | 0.040 |
| DeepSeek-V4-Pro | 0.337 | 0.740 | 0.077 | 0.800 | 0.901 | 0.680 | 0.850 | 0.150 | 0.680 | 0.160 | 0.000 | 0.000 | 0.480 | 0.260 | 0.000 |

## Recommendation

**continue_final_hardening**

- latest_only is weak on joint_revision_success as expected
- retrieve_all shows high overcitation_rate
- strong API models still struggle: Kimi-K2.6, GLM-5.1, DeepSeek-V4-Pro
- failure_diagnosis_accuracy is not trivially perfect

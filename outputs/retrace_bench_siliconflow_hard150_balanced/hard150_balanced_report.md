# ReTrace-Bench hard150_balanced — Final Hardening Report

Dataset: `data/retrace_bench_hard150_balanced/hard_150_en/scenarios.jsonl`
Scenarios: 150

## Gold expected_decision distribution (scheduled)

```json
{
  "use_current_memory": 75,
  "escalate": 12,
  "ask_clarification": 18,
  "refuse_due_to_policy": 15,
  "mark_unresolved": 30
}
```

| decision | count | share |
| --- | ---: | ---: |
| ask_clarification | 18 | 12.0% |
| escalate | 12 | 8.0% |
| mark_unresolved | 30 | 20.0% |
| refuse_due_to_policy | 15 | 10.0% |
| use_current_memory | 75 | 50.0% |

## SiliconFlow three-model metrics (balanced vs legacy hard150)

| model | metric | balanced | legacy (pre-schedule) | Δ |
| --- | --- | ---: | ---: | ---: |
| DeepSeek-V4-Pro | decision_macro_f1 | 0.195 | 0.285 | -0.090 |
| DeepSeek-V4-Pro | black_box_decision_accuracy | 0.513 | 0.747 | -0.233 |
| DeepSeek-V4-Pro | non_answer_decision_accuracy | 0.040 | 0.000 | +0.040 |
| DeepSeek-V4-Pro | memory_state_accuracy | 0.789 | 0.851 | -0.062 |
| DeepSeek-V4-Pro | evidence_f1 | 0.823 | 0.809 | +0.014 |
| DeepSeek-V4-Pro | failure_diagnosis_accuracy | 0.153 | 0.167 | -0.013 |
| DeepSeek-V4-Pro | joint_revision_success | 0.227 | 0.273 | -0.047 |
| DeepSeek-V4-Pro | format_failure_rate | 0.000 | 0.000 | +0.000 |
| GLM-5.1 | decision_macro_f1 | 0.342 | 0.665 | -0.323 |
| GLM-5.1 | black_box_decision_accuracy | 0.533 | 0.640 | -0.107 |
| GLM-5.1 | non_answer_decision_accuracy | 0.373 | 0.600 | -0.227 |
| GLM-5.1 | memory_state_accuracy | 0.791 | 0.849 | -0.058 |
| GLM-5.1 | evidence_f1 | 0.837 | 0.802 | +0.035 |
| GLM-5.1 | failure_diagnosis_accuracy | 0.233 | 0.400 | -0.167 |
| GLM-5.1 | joint_revision_success | 0.193 | 0.233 | -0.040 |
| GLM-5.1 | format_failure_rate | 0.000 | 0.000 | +0.000 |
| Kimi-K2.6 | decision_macro_f1 | 0.133 | 0.288 | -0.155 |
| Kimi-K2.6 | black_box_decision_accuracy | 0.500 | 0.760 | -0.260 |
| Kimi-K2.6 | non_answer_decision_accuracy | 0.000 | 0.000 | +0.000 |
| Kimi-K2.6 | memory_state_accuracy | 0.787 | 0.853 | -0.067 |
| Kimi-K2.6 | evidence_f1 | 0.694 | 0.661 | +0.033 |
| Kimi-K2.6 | failure_diagnosis_accuracy | 0.180 | 0.267 | -0.087 |
| Kimi-K2.6 | joint_revision_success | 0.133 | 0.133 | +0.000 |
| Kimi-K2.6 | format_failure_rate | 0.000 | 0.000 | +0.000 |

## Interpretation

- **memory_state_accuracy**: Should drop vs legacy hard150 when `is_distractor` is stripped from model prompts (no leakage).
- **non_answer_decision_accuracy**: More meaningful with balanced refuse/mark/ask/escalate shares (~50% non-answer cases).
- **joint_revision_success**: Headline stress metric; target &lt; 0.5 for all models.
- **format_failure_rate**: Must remain 0.0.

## Gates

- all_models_present: **PASS**
- format_failure_zero: **PASS**
- joint_below_half: **PASS**
- **overall hard150_balanced pass:** **PASS**

## hard500_candidate

Generated `data/retrace_bench_hard500_candidate/hard_500_en/scenarios.jsonl` (validator + gold oracle only).

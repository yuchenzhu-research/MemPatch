# ReTrace-Bench `calibration_80_en` (v1.0.0)

Small, clean, de-actionalized **smoke-test / quickstart** split of
ReTrace-Bench v1.0 (public split name: **`calibration`**).

> **Not** for headline benchmark claims, checkpoint selection, tuning, or model
> selection. Use `main_3000_en` for main results and `hard_300_en` for the
> stress evaluation. This split exists only to let you confirm an evaluation
> pipeline runs end-to-end in seconds.

- **Scenarios:** 80
- **Source type:** `controlled_synthetic`
- **Annotation status:** `synthetic_gold`
- **Benchmark version:** `1.0.0`

Decision-word leakage audit:
`scenarios_with_decision_word_leak = 0`
(`clean = true`).

## Regenerate

```bash
PYTHONPATH=. python scripts/generate_retrace_calibration_80.py
PYTHONPATH=. python scripts/validate_retrace_bench_dataset.py \
  --data data/retrace_bench/calibration_80_en/scenarios.jsonl
```

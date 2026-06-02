# Split Leakage Report: ReTrace-Bench HF Release

This release uses `data/test_800_templateheldout_en/` as the paper-facing
held-out benchmark split.

The repository split `data/retrace_bench/test_800_en/` is retained upstream as a
prototype/diagnostic split and should not be used as the final benchmark.

Command:

```bash
PYTHONPATH=. python scripts/check_retrace_split_leakage.py \
  --train data/retrace_supervision/train_3000_en/scenarios.jsonl \
  --dev data/retrace_supervision/dev_400_en/scenarios.jsonl \
  --test data/retrace_bench/test_800_templateheldout_en/scenarios.jsonl \
  --strict-template-heldout
```

Result: `OK`.

## Exact-Overlap Checks

- `scenario_id` overlap: none
- `memory_id` overlap: none
- `event_id` overlap: none
- exact public event text overlap: none
- exact `expected_answer` overlap: none

## Template-Level Checks

- train-to-test scenario signature overlap: `0` (`0.000`)
- dev-to-test scenario signature overlap: `0`
- train-to-test event-template overlap: `0`
- train-signature template lookup decision accuracy: `0.291`
- train-signature template lookup failure-mode accuracy: `0.091`

## Split Roles

- `supervision/train_3000_en/`: synthetic supervision pool, not a benchmark test.
- `supervision/dev_400_en/`: synthetic selection/dev pool, not a benchmark test.
- `data/test_800_templateheldout_en/`: held-out benchmark test split.

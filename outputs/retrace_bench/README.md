# ReTrace-Bench outputs

This directory holds model-suite prediction dumps and scored metrics for
ReTrace-Bench.

## Historical (pre-v1.0) outputs removed

The historical pre-v1.0 raw outputs (the `test800` / `first200` prediction
dumps and the `pilot_v2` / `full800` results, plus their logs) were **removed
from the current branch**. They described the legacy pre-v1.0 layout
(`test_800_templateheldout_en`, etc.), not the ReTrace-Bench v1.0 splits, and
carrying them forward would misrepresent old pilot numbers as current results.

They remain fully recoverable from the Git tag
`legacy-retrace-bench-pre-v1.0`.

## Official v1.0 outputs

Official ReTrace-Bench v1.0 model-suite outputs should be written under
`outputs/retrace_bench/v1_0/`, one subdirectory per split:

```text
outputs/retrace_bench/v1_0/main_3000/
outputs/retrace_bench/v1_0/hard_300/
outputs/retrace_bench/v1_0/calibration_80/
outputs/retrace_bench/v1_0/realistic_100_pending_annotation/
```

No official v1.0 results exist yet — the model suite will be re-run on the new
splits. Use the official evaluator to produce them, for example:

```bash
PYTHONPATH=. python scripts/evaluate_retrace_bench_predictions.py \
  --data data/retrace_bench/main_3000_en/scenarios.jsonl \
  --predictions path/to/predictions.jsonl \
  --out-metrics outputs/retrace_bench/v1_0/main_3000/my_model.metrics.json \
  --out-scored outputs/retrace_bench/v1_0/main_3000/my_model.scored.jsonl \
  --print-table
```

`realistic_100` results must **not** be treated as official until human
annotation is completed; until then they live under
`realistic_100_pending_annotation/` and `calibration_80` is smoke/quickstart
only (never for model selection or headline claims).

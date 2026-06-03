# ReTrace-Bench

ReTrace-Bench is the benchmark/resource paper track for agent memory revision
reliability. It is evaluation-only and method-neutral: any memory-enabled agent
can be scored with the same official schema, evaluator, metrics, and splits.

## ReTrace-Bench v1.0 splits

Four paper-facing splits (public names `main` / `hard` / `realistic` /
`calibration`):

| split | dir | size | role |
| --- | --- | --- | --- |
| `main` | `data/retrace_bench/main_3000_en/` | 3000 | controlled benchmark main split |
| `hard` | `data/retrace_bench/hard_300_en/` | 300 | long-context / multi-evidence stress split |
| `realistic` | `data/retrace_bench/realistic_100_en/` | 100 | realistic-style workflow split, annotation pending |
| `calibration` | `data/retrace_bench/calibration_80_en/` | 80 | smoke / quickstart only |

The legacy pre-v1.0 layout is recoverable from the Git tag
`legacy-retrace-bench-pre-v1.0`.

## Canonical Locations

- Benchmark package: `benchmark/retrace_bench/`
- Benchmark data: `data/retrace_bench/`
- ReTrace-Learn training/validation datasets: `data/retrace_learn/v1_0/`
- Benchmark docs/results: `docs/retrace_bench/`
- Paper workspace: `papers/retrace_bench/`
- Hugging Face release snapshot: `release/huggingface/ReTrace-Bench/`
- Examples: `examples/retrace_bench/`
- Tests: `tests/retrace_bench/`

## Quick Commands

```bash
PYTHONPATH=. python scripts/validate_retrace_bench_dataset.py \
  --data data/retrace_bench/main_3000_en/scenarios.jsonl

PYTHONPATH=. python scripts/evaluate_retrace_bench_predictions.py \
  --data data/retrace_bench/calibration_80_en/scenarios.jsonl \
  --predictions examples/retrace_bench/sample_predictions.jsonl \
  --print-table
```

For split roles, prediction schema, and public API usage, start with
[`benchmark/README.md`](benchmark/README.md).

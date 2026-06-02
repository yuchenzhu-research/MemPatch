# ReTrace-Bench

ReTrace-Bench is the benchmark/resource paper track for agent memory revision
reliability. It is evaluation-only and method-neutral: any memory-enabled agent
can be scored with the same official schema, evaluator, metrics, and splits.

## Canonical Locations

- Benchmark package: `benchmark/retrace_bench/`
- Benchmark data: `data/retrace_bench/`
- Supervision and selection pools: `data/retrace_supervision/`
- Benchmark docs/results: `docs/retrace_bench/`
- Paper workspace: `papers/retrace_bench/`
- Hugging Face release snapshot: `release/huggingface/ReTrace-Bench/`
- Examples: `examples/retrace_bench/`
- Tests: `tests/retrace_bench/`

## Quick Commands

```bash
PYTHONPATH=. python scripts/validate_retrace_bench_dataset.py \
  --data data/retrace_bench/test_800_templateheldout_en/scenarios.jsonl

PYTHONPATH=. python scripts/evaluate_retrace_bench_predictions.py \
  --data data/retrace_bench/test_800_templateheldout_en/scenarios.jsonl \
  --predictions examples/retrace_bench/sample_predictions.jsonl \
  --print-table
```

For split roles, prediction schema, and public API usage, start with
[`benchmark/README.md`](benchmark/README.md).

# ReTrace-Bench Generation And Audit Protocol

## Commands

```bash
python scripts/generate_retrace_bench_blueprints.py --count 1760 --out data/retrace_bench/blueprints_1760.jsonl --seed 42
python scripts/render_retrace_bench_dataset.py --blueprints data/retrace_bench/blueprints_1760.jsonl --out data/retrace_bench/stress_1760_en/scenarios.jsonl --renderer template --seed 42
python scripts/validate_retrace_bench_dataset.py --data data/retrace_bench/stress_1760_en/scenarios.jsonl
```

## Audit Rules

- Gold labels must originate from deterministic blueprints.
- Public input must not reveal hidden labels or benchmark-specific terminology.
- Evidence event IDs must exist in the rendered event trace.
- Memory IDs in expected state must appear in initial memory or be introduced by
  the hidden rubric's `introduced_memories` map.
- Cross-scope and distractor traps are tracked in metadata and checked by the
  validator.

## Baseline Runs

```bash
python scripts/run_retrace_bench_baseline.py --data data/retrace_bench/dev_800_en/scenarios.jsonl --baseline latest_only --out outputs/retrace_bench/latest_only_dev800.jsonl
python scripts/run_retrace_bench_baseline.py --data data/retrace_bench/dev_800_en/scenarios.jsonl --baseline retrieve_all --out outputs/retrace_bench/retrieve_all_dev800.jsonl
```

Each run writes predictions and a sibling `.metrics.json` summary.


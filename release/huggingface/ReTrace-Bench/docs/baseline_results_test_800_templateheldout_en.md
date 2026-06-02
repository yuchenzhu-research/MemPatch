# Offline Baseline Results: test_800_templateheldout_en

`data/retrace_bench/test_800_templateheldout_en/` is the candidate paper-facing held-out split. The existing `data/retrace_bench/test_800_en/` split is retained as prototype/diagnostic.

`data/retrace_supervision/train_3000_en/` and `data/retrace_supervision/dev_400_en/` are synthetic supervision/selection pools, not benchmark test sets. The template lookup diagnostic is a shortcut-leakage probe, not a deployable memory baseline.

Command:

```bash
PYTHONPATH=. python scripts/run_retrace_bench_ablation.py --data data/retrace_bench/test_800_templateheldout_en/scenarios.jsonl --out-dir outputs/retrace_bench/ablation_test_800_templateheldout_offline --max-cases 800
```

## Results

| group | baseline | oracle? | decision acc. | decision macro-F1 | non-answer acc. | key fact | memory state | evidence F1 | diagnosis | stale reuse |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sanity | latest_only | false | 0.291 | 0.090 | 0.000 | 0.107 | 0.323 | 0.107 | 0.091 | 0.000 |
| sanity | retrieve_all | false | 0.291 | 0.090 | 0.000 | 0.000 | 0.316 | 0.303 | 0.091 | 1.000 |
| memory_baseline | crud_memory | false | 0.496 | 0.357 | 0.321 | 0.694 | 0.502 | 0.694 | 0.229 | 0.000 |
| memory_baseline | heuristic_memory_state | false | 0.291 | 0.090 | 0.000 | 0.694 | 0.603 | 0.694 | 0.091 | 0.000 |
| memory_baseline | mem0_style | false | 0.292 | 0.140 | 0.034 | 0.694 | 0.393 | 0.694 | 0.203 | 0.000 |
| memory_baseline | rag_lexical | false | 0.291 | 0.090 | 0.000 | 0.007 | 0.316 | 0.305 | 0.091 | 0.993 |
| oracle | retrace_oracle_engine | true | 0.294 | 0.232 | 0.132 | 1.000 | 0.968 | 1.000 | 1.000 | 0.000 |

The oracle row is an upper-bound diagnostic for state/evidence/diagnosis paths, not a comparable deployable baseline.

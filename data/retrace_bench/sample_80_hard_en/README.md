# sample_80_hard_en

Hard calibration split for the general agent-memory reliability benchmark
(80 English scenarios, `retrace_bench_general_1` schema). This is a small
committed calibration fixture, **not** a full evaluation dataset.

It is deliberately compact but adversarial:

- multi-source event traces (7+ events with mixed trust levels and actors);
- distractor memories and cross-scope traps;
- stale-but-plausible notes that paraphrase the old answer;
- policy constraints, wrong-source attribution, hallucinated / false-premise
  claims, and forget / release / restore cases;
- a mix of direct answers and non-answer actions (escalate, ask_clarification,
  refuse_due_to_policy, mark_unresolved).

Coverage (all 8 domains and all 11 failure modes appear at least once):

| property | rate |
| --- | --- |
| 7+ events | 1.00 |
| 3+ memory entries | 1.00 |
| distractor memories | 0.60 |
| cross-scope traps | 0.70 |
| verified evidence vs. trusted-but-outdated note | 0.68 |
| rejects a false premise | 0.45 |
| requires a non-answer action | 0.35 |

## Regenerating

Deterministic; depends only on `--count`:

```bash
PYTHONPATH=. python scripts/generate_retrace_bench_hard.py --count 80
```

Validate:

```bash
PYTHONPATH=. python scripts/validate_retrace_bench_dataset.py \
  --data data/retrace_bench/sample_80_hard_en/scenarios.jsonl
pytest tests/retrace_bench/test_sample_80_hard_validation.py -q
```

`manifest.json` holds dataset metadata. The paper-facing held-out split is
`data/retrace_bench/test_800_templateheldout_en/`.

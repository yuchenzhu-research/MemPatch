# test_800_en

Held-out internal ReTrace-Bench evaluation set. No training, no prompt tuning, no policy optimization, no checkpoint selection.

## Provenance

- Schema: `retrace_bench_general_1` (general benchmark style).
- Scenario ID prefix: `rt-test-`.
- Deterministic seed range: `300000`-`300799`.
- Entity prefixes: case `CTE-`, project `PRTE-`, person `PTE-`, workspace `ws-te`.
- `training_targets` present: `false`.

This split shares no scenario, case, entity, memory ID, event ID, exact text, hidden gold, or seed range with the other splits (see `docs/retrace_bench/split_leakage_report.md`).

## Model input vs. evaluation

- Model input: `workflow_context`, `public_input`, `tasks` only.
- Evaluation may read `hidden_gold`.

## Held-out policy

`test_800_en` is the internal ReTrace-Bench held-out evaluation set: **no** training, **no** prompt tuning, **no** policy optimization, and **no** checkpoint selection may use it.

## Coverage

- Scenarios: 800
- Domains: 8/8; failure modes: 11/11.
- Decisions: {'ask_clarification': 73, 'escalate': 73, 'mark_unresolved': 73, 'refuse_due_to_policy': 73, 'use_current_memory': 508}
- Rates: {'events_ge_7': 1.0, 'memories_ge_3': 1.0, 'distractors': 0.6, 'cross_scope': 0.7, 'verified_over_trusted': 0.6813, 'false_premise': 0.455, 'non_answer': 0.365}

## Regenerate

```bash
PYTHONPATH=. python scripts/generate_retrace_data_splits.py --train-count 3000 --dev-count 400 --test-count 800
```

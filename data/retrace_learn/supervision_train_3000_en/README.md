# train_3000_en

Synthetic supervision pool for future ReTrace-Learn open-model typed revision proposer training (Graph Extractor -> Typed Revision Proposer -> Authorization Court). This is **not** a benchmark and must never be scored as held-out evaluation.

## Provenance

- Schema: `retrace_bench_general_1` (general benchmark style).
- Scenario ID prefix: `rt-train-`.
- Deterministic seed range: `100000`-`102999`.
- Entity prefixes: case `CTR-`, project `PRTR-`, person `PTR-`, workspace `ws-tr`.
- `training_targets` present: `true`.

This split shares no scenario, case, entity, memory ID, event ID, exact text, hidden gold, or seed range with the other splits (see `docs/retrace_bench/split_leakage_report.md`).

## Model input vs. evaluation

- Model input: `workflow_context`, `public_input`, `tasks` only.
- Evaluation may read `hidden_gold`.
- `training_targets` (typed revision actions, target memory state, supporting evidence, optional evidence graph) is for future ReTrace-Learn method training and selection. It is **not** model input at evaluation time.

## Coverage

- Scenarios: 3000
- Domains: 8/8; failure modes: 11/11.
- Decisions: {'ask_clarification': 273, 'escalate': 273, 'mark_unresolved': 273, 'refuse_due_to_policy': 273, 'use_current_memory': 1908}
- Rates: {'events_ge_7': 1.0, 'memories_ge_3': 1.0, 'distractors': 0.6, 'cross_scope': 0.7, 'verified_over_trusted': 0.6817, 'false_premise': 0.4547, 'non_answer': 0.364}

## Regenerate

```bash
PYTHONPATH=. python scripts/generate_retrace_data_splits.py --train-count 3000 --dev-count 400 --test-count 800
```

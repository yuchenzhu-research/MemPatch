# dev_400_en

Development / selection split for prompt selection, policy selection, checkpoint selection, and validation-gated prompt/policy edits.

## Provenance

- Schema: `retrace_bench_general_1` (general benchmark style).
- Scenario ID prefix: `rt-dev-`.
- Deterministic seed range: `200000`-`200399`.
- Entity prefixes: case `CDV-`, project `PRDV-`, person `PDV-`, workspace `ws-dv`.
- `training_targets` present: `true`.

This split shares no scenario, case, entity, memory ID, event ID, exact text, hidden gold, or seed range with the other splits (see `docs/retrace_bench/split_leakage_report.md`).

## Model input vs. evaluation

- Model input: `workflow_context`, `public_input`, `tasks` only.
- Evaluation may read `hidden_gold`.
- `training_targets` (typed revision actions, target memory state, supporting evidence, optional evidence graph) is for future ReTrace-Learn method training and selection. It is **not** model input at evaluation time.

## Coverage

- Scenarios: 400
- Domains: 8/8; failure modes: 11/11.
- Decisions: {'ask_clarification': 36, 'escalate': 36, 'mark_unresolved': 37, 'refuse_due_to_policy': 36, 'use_current_memory': 255}
- Rates: {'events_ge_7': 1.0, 'memories_ge_3': 1.0, 'distractors': 0.6, 'cross_scope': 0.7, 'verified_over_trusted': 0.6825, 'false_premise': 0.455, 'non_answer': 0.3625}

## Regenerate

```bash
PYTHONPATH=. python scripts/generate_retrace_data_splits.py --train-count 3000 --dev-count 400 --test-count 800
```

# ReTrace-Bench prediction example

This directory shows how to score predictions with the official ReTrace-Bench
evaluator. The evaluator runs no model and needs no API keys — it only scores an
existing predictions file against a scenarios file.

## Install

The benchmark method package (`retracemem`) is pip-installable; the
evaluation-only `benchmark.retrace_bench` package is imported via `PYTHONPATH`:

```bash
pip install -e ".[dev]"
# benchmark.retrace_bench is not installed by pip; run benchmark commands with:
export PYTHONPATH=.
```

So every benchmark command below is prefixed with `PYTHONPATH=.`.

## Evaluate the sample predictions

```bash
PYTHONPATH=. python scripts/evaluate_retrace_bench_predictions.py \
  --data data/retrace_bench/sample_80_hard_en/scenarios.jsonl \
  --predictions examples/retrace_bench/sample_predictions.jsonl \
  --out-metrics outputs/retrace_bench/sample_predictions.metrics.json \
  --out-scored outputs/retrace_bench/sample_predictions.scored.jsonl \
  --print-table
```

`--data` accepts either a `scenarios.jsonl` file or a directory containing one.
For the canonical paper-facing benchmark, point `--data` at
`data/retrace_bench/test_800_templateheldout_en/`.

You can also call the Python API directly:

```python
from benchmark.retrace_bench.api import (
    load_scenarios, load_predictions, evaluate_predictions,
)

scenarios = load_scenarios("data/retrace_bench/sample_80_hard_en")
predictions = load_predictions("examples/retrace_bench/sample_predictions.jsonl")
result = evaluate_predictions(scenarios, predictions, strict=True)
print(result["headline_metrics"])
```

## Prediction schema

One JSON object per line. Match each scenario by `scenario_id`. Two equivalent
formats are accepted.

Canonical (nested `response`):

```json
{
  "scenario_id": "rb-hard-en-00001",
  "response": {
    "answer": "C-2000 should follow the updated release blocker path verified for PROJ-A11.",
    "decision": "use_current_memory",
    "memory_state": {
      "m-rb-hard-en-00001-target": "outdated",
      "m-rb-hard-en-00001-replacement": "current"
    },
    "evidence_event_ids": ["e-rb-hard-en-00001-06"],
    "failure_diagnosis": "stale_memory_reuse"
  }
}
```

Flat (response fields at top level) is also accepted:

```json
{
  "scenario_id": "rb-hard-en-00001",
  "answer": "...",
  "decision": "use_current_memory",
  "memory_state": {"m-rb-hard-en-00001-target": "outdated"},
  "evidence_event_ids": ["e-rb-hard-en-00001-06"],
  "failure_diagnosis": "stale_memory_reuse"
}
```

### Strict field semantics

- **`decision`** (required): exactly one of the five revision-decision labels:
  `use_current_memory`, `escalate`, `ask_clarification`, `refuse_due_to_policy`,
  `mark_unresolved`.
- **`memory_state`**: map of `memory_id -> status`, where each status is one of
  the eight labels: `current`, `outdated`, `blocked`, `unresolved`,
  `out_of_scope`, `deleted`, `should_not_store`, `restored`.
- **`evidence_event_ids`**: list of `event_id` values drawn from the scenario's
  `public_input.event_trace`. IDs not present in the trace are rejected.
- **`failure_diagnosis`**: one of the eleven failure-mode labels:
  `stale_memory_reuse`, `under_update`, `over_update`, `conflict_collapse`,
  `scope_leakage`, `policy_violation`, `wrong_source_attribution`,
  `memory_hallucination`, `unnecessary_memory_write`, `failure_to_forget`,
  `failure_to_release_or_restore`.
- **`answer`**: free text answer for the black-box task.

In `--strict` mode (default), missing/duplicate/extra predictions and invalid
labels or evidence IDs raise an error. Use `--no-strict` (or `--allow-missing`)
to score whatever is valid and collect the rest as warnings/errors.

The `sample_predictions.jsonl` here is a small, complete, runnable submission
for the 80-scenario `sample_80_hard_en` calibration split (so the strict command
above scores cleanly). It is only an illustration — it is **not** a submission
for the canonical 800-scenario `test_800_templateheldout_en` benchmark split.

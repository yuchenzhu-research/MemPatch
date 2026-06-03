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
  --data data/retrace_bench/calibration_80_en/scenarios.jsonl \
  --predictions examples/retrace_bench/sample_predictions.jsonl \
  --out-metrics outputs/retrace_bench/sample_predictions.metrics.json \
  --out-scored outputs/retrace_bench/sample_predictions.scored.jsonl \
  --print-table
```

`--data` accepts either a `scenarios.jsonl` file or a directory containing one.
For the primary paper-facing benchmark, point `--data` at
`data/retrace_bench/main_3000_en/` (or `hard_300_en/` for the stress split).

You can also call the Python API directly:

```python
from benchmark.retrace_bench.api import (
    load_scenarios, load_predictions, evaluate_predictions,
)

scenarios = load_scenarios("data/retrace_bench/calibration_80_en")
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
  "scenario_id": "rb-cal-en-000001",
  "response": {
    "answer": "<free-text answer for the black-box task>",
    "decision": "use_current_memory",
    "memory_state": {
      "m-rb-cal-en-000001-target": "current",
      "m-rb-cal-en-000001-distractor": "out_of_scope"
    },
    "evidence_event_ids": ["e-rb-cal-en-000001-08"],
    "failure_diagnosis": "under_update"
  }
}
```

Flat (response fields at top level) is also accepted:

```json
{
  "scenario_id": "rb-cal-en-000001",
  "answer": "...",
  "decision": "use_current_memory",
  "memory_state": {"m-rb-cal-en-000001-target": "current"},
  "evidence_event_ids": ["e-rb-cal-en-000001-08"],
  "failure_diagnosis": "under_update"
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
  `failure_to_release_or_restore`. A value that is neither a canonical label nor
  a documented normalized alias is rejected in strict mode.
- **`answer`**: free text answer for the black-box task.

In `--strict` mode (default), the following raise an error:
missing/duplicate/extra predictions, an unknown `decision` label, an unknown
`memory_state` status, an `evidence_event_ids` entry absent from the scenario's
`event_trace`, and an unrecognized `failure_diagnosis` label. Omitting a status
for some visible `initial_memory` IDs is reported as a **warning only** (it does
not block strict scoring), so partial `memory_state` maps are still scored. Use
`--no-strict` (or `--allow-missing`) to downgrade all errors to collected
warnings/errors and score whatever is valid.

The `sample_predictions.jsonl` here is a small, complete, runnable submission
for the 80-scenario `calibration_80_en` smoke/quickstart split (so the strict
command above scores cleanly). It is only an illustration — it is **not** a
submission for the paper-facing `main_3000_en` / `hard_300_en` splits, and the
`calibration` split must not be used for model selection or headline claims.

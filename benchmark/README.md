# MemPatch-Bench Package

`benchmark/retrace_bench/` contains the MemPatch-Bench evaluator/API code. The `retrace_bench` package path is retained for compatibility.

MemPatch-Bench is the evaluation layer of the unified MemPatch paper system. It stays method-neutral: it scores predictions against gold labels but does not implement proposal or authorization logic.

Surviving code should be limited to:

- public scoring API
- prediction/schema normalization
- taxonomy
- scorers and aggregate metrics
- public-view sanitization
- dataset/prediction validation helpers
- small utility modules required by those surfaces

Generation, audit, report, pilot, and paper-writing code does not belong in this package.

## Optional Model Runner

`benchmark/retrace_bench/model_runner.py` is an optional local runner for
producing prediction files. It is separate from the official scoring API and
never sends `hidden_gold` to providers; it prompts with the sanitized
`public_scenario_view(...)` only.

Supported providers:

- `openai`
- `google` / `gemini`
- `anthropic`
- `openai_compatible`
- `siliconflow`
- `deepseek`
- `minimax`

Public CLI:

```bash
python scripts/run_retrace_bench_model.py \
  --data local/ReTrace-Bench/calibration/scenarios.jsonl \
  --provider siliconflow \
  --model deepseek-ai/DeepSeek-V3 \
  --out-predictions local/predictions/siliconflow_calibration.jsonl \
  --max-cases 10 \
  --resume
```

## Prediction JSONL Schema

The evaluator accepts one JSON object per line:

```json
{"scenario_id":"scenario-id","response":{"answer":"...","decision":"use_current_memory","memory_state":{"memory-id":"current"},"evidence_event_ids":["event-id"],"failure_diagnosis":"stale_memory_reuse"}}
```

Canonical `response` fields:

- `answer`: free-text answer for the black-box task.
- `decision`: one of `use_current_memory`, `escalate`, `ask_clarification`, `refuse_due_to_policy`, `mark_unresolved`.
- `memory_state`: object mapping visible `memory_id` values to one of `current`, `outdated`, `blocked`, `unresolved`, `out_of_scope`, `deleted`, `should_not_store`, `restored`.
- `evidence_event_ids`: list of visible `event_id` values that support the response.
- `failure_diagnosis`: one of `stale_memory_reuse`, `under_update`, `over_update`, `conflict_collapse`, `scope_leakage`, `policy_violation`, `wrong_source_attribution`, `memory_hallucination`, `unnecessary_memory_write`, `failure_to_forget`, `failure_to_release_or_restore`.

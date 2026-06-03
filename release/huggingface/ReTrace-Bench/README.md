---
license: cc-by-4.0
language:
- en
pretty_name: ReTrace-Bench
task_categories:
- question-answering
- text-classification
- text-generation
tags:
- agent-memory
- llm-agents
- benchmark
- memory-revision
- long-context
- reliability
- evaluation
configs:
- config_name: default
  data_files:
  - split: main
    path: main/main_3000_en.jsonl
  - split: hard
    path: hard/hard_300_en.jsonl
  - split: realistic
    path: realistic/realistic_100_en.jsonl
  - split: calibration
    path: calibration/calibration_80_en.jsonl
---

# ReTrace-Bench

ReTrace-Bench v1.0.0 evaluates **agent memory revision
reliability**: whether a system can process new evidence to update, block,
release, reaffirm, or reject memory states without introducing stale,
out-of-scope, or policy-invalid memory. It is not merely a final-answer
benchmark — coarse decision accuracy can overestimate memory reliability, so the
benchmark also scores memory-state tracking, evidence grounding, and failure
diagnosis.

## 1. Dataset Summary

Four paper-facing splits, all English, controlled or realistic-style synthetic,
constructed with a leakage-audited (de-actionalized) procedure: authoritative
records never contain a decision-action phrase, so the correct revision decision
must be recovered by reasoning over described state rather than string matching.

## 2. Split Overview

| split | size | role |
|-------|------|------|
| `main` | 3000 | controlled benchmark main split |
| `hard` | 300 | long-context and multi-evidence stress split |
| `realistic` | 100 | realistic-style workflow split, annotation pending |
| `calibration` | 80 | smoke/quickstart only |

## 3. Task Definition

Each scenario presents an initial memory set and a chronological event trace.
The system must decide how memory should be revised and answer four task views:
black-box answer, memory-state classification, evidence retrieval, and failure
diagnosis.

## 4. Scenario Schema

Source-of-truth scenarios are nested JSON objects with `scenario_id`, `split`,
`domain`, `primary_failure_mode`, `difficulty`, `workflow_context`,
`public_input` (`initial_memory`, `event_trace`), `tasks`, `hidden_gold`, and
`metadata`. So the Hugging Face viewer can render every column, nested fields are published as
JSON string columns (`public_input_json`, `tasks_json`, `hidden_gold_json`,
`metadata_json`, `secondary_failure_modes_json`); parse them with
`json.loads(...)`.

## 5. Prediction Schema

One JSON object per line, matched to scenarios by `scenario_id`:

```json
{
  "scenario_id": "<scenario id>",
  "response": {
    "answer": "<free-text answer>",
    "decision": "use_current_memory",
    "memory_state": {"<memory_id>": "outdated"},
    "evidence_event_ids": ["<event_id from public_input.event_trace>"],
    "failure_diagnosis": "stale_memory_reuse"
  }
}
```

- `decision`: one of `use_current_memory`, `escalate`, `ask_clarification`,
  `refuse_due_to_policy`, `mark_unresolved`.
- `memory_state`: `memory_id -> status` (`current`, `outdated`, `blocked`,
  `unresolved`, `out_of_scope`, `deleted`, `should_not_store`, `restored`).
- `evidence_event_ids`: `event_id` values from `public_input.event_trace`.
- `failure_diagnosis`: one of the eleven failure-mode labels.

## 6. Official Evaluator

ReTrace-Bench ships an official scorer that runs no model and needs no API keys.
Clone the repository at https://github.com/yuchenzhu-research/ReTrace, then score a predictions file:

```bash
PYTHONPATH=. python scripts/evaluate_retrace_bench_predictions.py \
  --data data/retrace_bench/main_3000_en/scenarios.jsonl \
  --predictions path/to/predictions.jsonl \
  --out-metrics outputs/retrace_bench/my_model.metrics.json \
  --out-scored outputs/retrace_bench/my_model.scored.jsonl \
  --print-table
```

See `examples/retrace_bench/` for a runnable example and the Python API
(`benchmark.retrace_bench.api`).

## 7. Metrics

Primary metrics: `decision_macro_f1`, `non_answer_decision_accuracy`,
`memory_state_accuracy`, `evidence_f1`, `failure_diagnosis_accuracy`,
`stale_reuse_rate`.

## 8. Benchmark Hygiene / Leakage Audit

Every split passes a decision-word leakage audit: no verified/trusted
(authoritative) record contains a decision-action phrase tied to one of the five
gold decisions. Scenario, memory, and event IDs are disjoint across splits, and
there is no universal cross-scope distractor shortcut.

## 9. Annotation Status

- `main`, `hard`, `calibration`: `controlled_synthetic`, synthetic gold.
- `realistic`: `realistic_style_synthetic`, **`annotation_status = pending`**.
  Its `hidden_gold` fields are intentionally empty; human annotation will be
  added later via `annotations/realistic_100_template.jsonl`. No human validation
  is claimed and no public-source provenance is claimed.

## 10. Intended Use

`main` is for primary benchmark results; `hard` for long-context / multi-evidence
stress; `realistic` for realistic workflow texture once annotated. `calibration`
is a smoke/quickstart split only: **it is not a model-selection / checkpoint-selection validation set and must not be used to tune or select systems**, and it must not be used for headline claims.

## 11. Limitations

`main` / `hard` / `calibration` gold is synthetic. `realistic` is unannotated in
this release. The legacy pre-v1.0 layout is not part of this release and is
recoverable only from the Git tag `legacy-retrace-bench-pre-v1.0`.

## 12. Citation

```bibtex
@misc{retrace_bench,
  title        = {ReTrace-Bench: Evaluating Agent Memory Revision Reliability},
  author       = {ReTrace-Bench Authors},
  year         = {2026},
  howpublished = {\url{https://github.com/yuchenzhu-research/ReTrace}},
  note         = {Benchmark version 1.0.0, commit 95b14f86a58fda15fc0e8d468d965932d76f1e89}
}
```

## 13. License

Distributed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE) license.

*Total packaged scenarios: 3480.*

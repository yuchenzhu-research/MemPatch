# MemPatch-Bench

MemPatch-Bench is the evaluation layer for Rapid Memory Integration (RMI): measuring how reliably LLM agents integrate new evidence with prior memory states through typed revision actions.

Keep in GitHub:

- `benchmark/retrace_bench/`: public scoring API, taxonomy, schema helpers, scorers, public-view sanitization, validation helpers, and minimal evaluator support.
- `hf_release/retrace_bench_v1_1/`: latest Hugging Face release metadata/package snapshot.
- `tests/retrace_bench/`: minimal tests for the surviving evaluator/API surface.

Do not keep local generated reports, paper drafts, sample files, run dumps, annotation packets, or duplicate benchmark-data copies.

Official public data lives on Hugging Face: `Sylvan-Vale-Moon/ReTrace-Bench`.

## Paper-facing response fields

Predictions expose a `response` object scored against `hidden_gold`:

- `decision` — revision decision
- `memory_state` — per-memory usability labels
- `evidence_event_ids` — visible supporting events
- `failure_diagnosis` — diagnostic failure mode

DPA authorization inside the MemPatch scaffold yields belief statuses that align with `memory_state` labels.

# MemPatch repository inventory

Unified MemPatch paper system. Implementation paths: `benchmark/retrace_bench/`, `src/retrace_learn/`, `src/retracemem/`.

## Layout

| Path | Role |
|------|------|
| `README.md`, `AGENTS.md` | Active authority |
| `benchmark/retrace_bench/` | MemPatch-Bench evaluator API, taxonomy, scorers |
| `hf_release/retrace_bench_v1_1/` | HF release metadata |
| `src/retrace_learn/` | Scenario View Builder, Revision Response Policy, feedback |
| `src/retracemem/` | DPA, `authorize`, memory store, multi-agent commit |
| `scripts/` | evaluate, validate, run model |
| `data/retrace_learn/` | Method-side splits + manifest |
| `docs/` | Paper outline and archived planning notes |
| `local/` | Gitignored downloads, predictions, results |

## Paper interface

Benchmark `response` fields: `decision`, `memory_state`, `evidence_event_ids`, `failure_diagnosis`, `answer`.

Gold: `hidden_gold` with canonical v1.1 fields only.

## Verification

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q benchmark scripts src
```

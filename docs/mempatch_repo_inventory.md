# MemPatch repository inventory

**Algorithm spec:** `docs/mempatch_revision_module.md`

## Layout

| Path | MemPatch Revision Module role |
|------|-------------------------------|
| `benchmark/retrace_bench/` | MemPatch-Bench evaluator |
| `src/retrace_learn/runtime/graph_extractor.py` | Scenario View Builder |
| `src/retrace_learn/runtime/learned_proposer.py` | Revision Response Policy |
| `src/retrace_learn/runtime/dpa_runtime.py` | Parse + projection front-end |
| `src/retracemem/authorization.py` | DPA-Consistent Projection kernel |
| `src/retrace_learn/runtime/reward.py` | Benchmark-grounded Feedback |
| `scripts/` | CLI: run, evaluate, validate |

# MemPatch Agent Instructions

**Read order:** `AGENTS.md` → `README.md` → `docs/mempatch_revision_module.md`

Blind-review artifact: do not add venue names, author identity, or personal repository URLs to public-facing docs.

## Unified paper

**MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents**

One story:

- **MemPatch-Bench** — paper-facing `response` interface and `hidden_gold` scoring
- **MemPatch Revision Module** — algorithm module for benchmark-compatible revision responses
- **DPA** — deterministic verifier inside the module (`authorize`)
- **Benchmark-grounded feedback** — decomposable reward interface in `reward.py` (not wired to a training loop in this artifact)

## MemPatch Revision Module

```text
V ← BuildScenarioRevisionView(S, M)
r_raw ← πθ(V)
a ← ParseRevisionResponse(r_raw)
T ← DPAConsistentProjection(A, a, V)
r_final ← ProjectToBenchmarkResponse(T, r_raw)
```

Internal roles (not paper contributions):

1. Scenario View Builder — `graph_extractor.py`, `scenario_revision.py`
2. Revision Response Policy — `learned_proposer.py`
3. DPA-Consistent Projection — `dpa_runtime.py`, `benchmark_projection.py`, `authorization.py`
4. Benchmark-grounded Feedback — `reward.py`

Public evaluator import: `benchmark.mempatch_bench.api`

## Benchmark response interface

`response.decision`, `response.memory_state`, `response.evidence_event_ids`, `response.failure_diagnosis`, `response.answer`

Gold: canonical v1.1 `hidden_gold` fields only.

## DPA

The model proposes; DPA authorizes; the benchmark evaluates `memory_state`. Call only `authorize(...)`.

## Baselines (internal config names)

- Typed-action baseline over fixed revision view → DPA projection
- DirectJudge baseline — bypasses projection
- Full Revision Module — `run_mempatch_revision_module.py`

## Verification

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q benchmark scripts src tests
PYTHONPATH=.:src .venv/bin/python -m pytest -q
```

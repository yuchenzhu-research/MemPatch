# MemPatch Agent Instructions

**Read order:** `AGENTS.md` → `README.md`

Blind-review artifact: do not add venue names, author identity, or personal repository URLs to public-facing docs.

## Pre-commit hygiene

Before every commit, delete Python cache artifacts (never commit these):

```bash
rm -rf .pycache_compile .pytest_cache
find benchmark scripts src tests -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
```

Do **not** `git add local/` — all generated models, adapters, and scratch data stay gitignored.

## Unified paper

**MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents**

- **MemPatch-Bench** — `response` interface + `hidden_gold` scoring (`benchmark/api.py`)
- **MemPatch Revision Module** — `src/mempatch_learn/` (view → policy → DPA → projection)
- **DPA** — deterministic verifier (`mempatch_dpa.authorize`)

Public evaluator: `from benchmark.api import evaluate_predictions, load_scenarios`

## Dataset splits (v1.3)

| Split | Full size | Role |
|-------|----------:|------|
| `train` | 2700 | SFT only |
| `validation` | 800 | Dev eval |
| `test` | 500 | Held-out final eval |

Generate: `scripts/generate_mempatch.py --full --out-dir hf_release/mempatch`  
Audit gate: `scripts/audit_decision_boundary.py` (must pass before training)

## Revision module pipeline

```text
V ← build_scenario_revision_view(S, M)     # scenario_revision.py
r_raw ← πθ(V)                              # learned_proposer.py
T ← DPAConsistentProjection(A, a, V)       # dpa_runtime.py
r_final ← project_to_benchmark_response    # benchmark_projection.py
```

## Benchmark response interface

`response.decision`, `response.memory_state`, `response.evidence_event_ids`, `response.failure_diagnosis`, `response.answer`

Gold: canonical `hidden_gold` fields via `benchmark.general_taxonomy.canonical_hidden_gold_fields`.

## DPA

The model proposes; DPA authorizes; the benchmark evaluates `memory_state`. Call only `authorize(...)`.

## Baselines

- Typed-action + DPA projection — `run_mempatch_revision_module.py --policy scripted`
- DirectJudge — `run_mempatch_model.py`
- Full module — `run_mempatch_revision_module.py --policy prompt`

## Verification

```bash
rm -rf .pycache_compile .pytest_cache
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q benchmark scripts src tests
PYTHONPATH=.:src .venv/bin/python -m pytest -q
PYTHONPATH=.:src .venv/bin/python scripts/audit_decision_boundary.py \
  --data hf_release/mempatch/train --data hf_release/mempatch/validation --data hf_release/mempatch/test
```

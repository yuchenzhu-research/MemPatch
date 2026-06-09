# MemPatch Agent Instructions

**Read order:** `AGENTS.md` → `README.md`

Blind-review artifact: do not add venue names, author identity, or personal repository URLs to public-facing docs.

## Pre-commit hygiene

Before every commit, delete Python cache artifacts (never commit these):

```bash
rm -rf .pycache_compile .pytest_cache
find benchmark scripts src tests -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
```

The `find` covers all `scripts/` subdirectories (`eval/`, `workflows/`, `data/`, `mlx/`, `memory/`, `analysis/`).

Do **not** `git add local/` — all generated models, adapters, and scratch data stay gitignored.

## Responsibility boundaries

| Package | Role | Does **not** |
|---------|------|--------------|
| `benchmark/` | Dataset format, `public_view`, **`evaluate_predictions` scoring** | Implement revision algorithm or DPA |
| `benchmark/model_runner.py` | Optional JSON prompt adapter (`build_prompt`) for baselines | Revision view, proposer, or projection |
| `src/mempatch_learn/` | Path A/B method: view → proposer → DPA → projection | Define or compute benchmark metrics |
| `src/mempatch_dpa/` | Deterministic `authorize` kernel | Score predictions; only used inside revision module |

Public evaluator: `from benchmark.api import evaluate_predictions, load_scenarios`

Data flow: `scenarios.jsonl` → runner (`scripts/eval/`) → `predictions.jsonl` → `benchmark.api` → metrics.

## Unified paper

**MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents**

- **MemPatch-Bench** — `response` interface + `hidden_gold` scoring (`benchmark/api.py`)
- **MemPatch Revision Module** — `src/mempatch_learn/` (view → policy → DPA → projection)
- **DPA** — deterministic verifier (`mempatch_dpa.authorize`)

## Dataset splits (v1.3)

| Split | Full size | Role |
|-------|----------:|------|
| `train` | 2700 | SFT only |
| `validation` | 800 | Dev eval |
| `test` | 500 | Held-out final eval |

Generate: `scripts/data/generate_mempatch.py --full --out-dir hf_release/mempatch`  
Audit gate: `scripts/workflows/audit_decision_boundary.py` (must pass before training)

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

## Baselines and experiment scripts

| Line | Script |
|------|--------|
| External memory (RAG / full / mem0 / base) | `scripts/eval/run_mempatch_memory_baselines.py` |
| DirectJudge | `scripts/eval/run_mempatch_model.py` |
| Path A (typed actions + DPA) | `scripts/eval/run_mlx_revision_module_eval.py` |
| Path B (LoRA, direct JSON) | `scripts/eval/run_mlx_lora_smoke_eval.py` |
| Revision module (scripted/prompt) | `scripts/eval/run_mempatch_revision_module.py --policy scripted\|prompt` |
| Score existing predictions | `scripts/workflows/evaluate_mempatch_predictions.py` |
| Paper pipeline | `scripts/workflows/run_paper_pipeline.sh` |

RAG/full baselines: `scripts/memory/mempatch_memory_context.py` filters context, then `benchmark.model_runner.build_prompt` embeds `required_output_schema` — model output must be JSON.

## Local workspace

Recommended under `local/` (gitignored): `models/`, `adapters/`, `data/`, `runs/{paper,baselines}/`, `logs/`.

## Verification

```bash
rm -rf .pycache_compile .pytest_cache
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q benchmark scripts src tests
PYTHONPATH=.:src .venv/bin/python -m pytest -q
PYTHONPATH=.:src .venv/bin/python scripts/workflows/audit_decision_boundary.py \
  --data hf_release/mempatch/train --data hf_release/mempatch/validation --data hf_release/mempatch/test
```

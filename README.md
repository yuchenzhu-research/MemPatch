# MemPatch

**MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents**

## Problem

RMI (Rapid Memory Integration): when an LLM agent receives new evidence, it must determine whether each memory is `current`, `outdated`, `blocked`, `unresolved`, `out_of_scope`, or `restored` — not merely append new information.

MemPatch turns memory revision into a **constrained benchmark-compatible state-transition problem** rather than free-form answer generation.

## One paper, one module

| Piece | Role |
|-------|------|
| **MemPatch-Bench** | Defines scenarios, `hidden_gold`, and the paper-facing `response` interface |
| **MemPatch Revision Module** | Produces benchmark-compatible revision responses |
| **DPA** (`authorize(...)`) | Deterministic verifier inside the module |
| **Benchmark-grounded feedback** | Training signal from benchmark metrics |

Algorithm spec: [`docs/mempatch_revision_module.md`](docs/mempatch_revision_module.md)

Experiment plan: [`docs/experiments_open_source_budget_plan.md`](docs/experiments_open_source_budget_plan.md)

## MemPatch Revision Module (summary)

```text
V  ← BuildScenarioRevisionView(scenario, memories)
r_raw ← RevisionResponsePolicy(V)
T  ← DPAConsistentProjection(A, parse(r_raw))
r_final ← ProjectToBenchmarkResponse(T, r_raw)
```

## Paper-facing response

```json
{
  "decision": "use_current_memory",
  "memory_state": {"m1": "current", "m2": "outdated"},
  "evidence_event_ids": ["e2", "e5"],
  "failure_diagnosis": "stale_memory_reuse",
  "answer": "..."
}
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Evaluate predictions

```bash
PYTHONPATH=. python scripts/evaluate_mempatch_predictions.py \
  --data path/to/scenarios.jsonl \
  --predictions path/to/predictions.jsonl \
  --print-table
```

```python
from benchmark.mempatch_bench.api import load_scenarios, load_predictions, evaluate_predictions

result = evaluate_predictions(
    load_scenarios("path/to/scenarios.jsonl"),
    load_predictions("path/to/predictions.jsonl"),
    strict=True,
)
```

Fixture smoke (no model cost):

```bash
PYTHONPATH=. python scripts/evaluate_mempatch_predictions.py \
  --data tests/fixtures/smoke_scenarios.jsonl \
  --predictions tests/fixtures/smoke_predictions.jsonl \
  --print-table
```

## MemPatch Revision Module (method path)

```bash
python scripts/run_mempatch_revision_module.py \
  --data local/MemPatch/main/scenarios.jsonl \
  --out-predictions local/predictions/mempatch_main20.jsonl \
  --max-cases 20 \
  --resume
```

## Direct Response baseline

```bash
python scripts/run_mempatch_model.py \
  --data local/MemPatch/main/scenarios.jsonl \
  --provider siliconflow \
  --model <OPEN_MODEL_NAME> \
  --out-predictions local/predictions/direct_main20.jsonl \
  --max-cases 20 \
  --resume
```

Prompt size smoke (no API call):

```bash
python scripts/run_mempatch_model.py \
  --data tests/fixtures/smoke_scenarios.jsonl \
  --provider siliconflow \
  --model <OPEN_MODEL_NAME> \
  --out-predictions local/predictions/smoke.jsonl \
  --max-cases 1 \
  --print-prompt-stats
```

Public evaluation data: Hugging Face artifact `MemPatch` v1.1 (`main`: 3000, `hard`: 500). See `hf_release/mempatch_v1_1/manifest.json`. Download JSONL into `local/MemPatch/` (not vendored in git).

## Verification

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q benchmark scripts src tests
PYTHONPATH=.:src .venv/bin/python -m pytest -q
```

## Citation (anonymous placeholder)

```bibtex
@misc{mempatch2026,
  title  = {MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents},
  author = {Anonymous},
  year   = {2026},
  note   = {Evaluation-only benchmark release.}
}
```

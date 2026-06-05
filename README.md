# MemPatch

**MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents**

## Problem

RMI (Rapid Memory Integration): when an LLM agent receives new evidence, it must determine whether each memory is `current`, `outdated`, `blocked`, `unresolved`, `out_of_scope`, or `restored` — not merely append new information.

MemPatch turns memory revision into a **constrained benchmark-compatible state-transition problem** rather than free-form answer generation.

## One paper, one module

| Piece | Role |
|-------|------|
| **MemPatch-Bench** | Defines scenarios, `hidden_gold`, and the paper-facing `response` interface |
| **MemPatch Revision Module** | Learns to produce benchmark-compatible revision responses |
| **DPA** (`authorize(...)`) | Deterministic verifier inside the module: model proposes, DPA authorizes, benchmark evaluates |
| **Benchmark-grounded feedback** | Improves the Revision Response Policy from `memory_state` / evidence / diagnostic metrics |

Algorithm spec: [`docs/mempatch_revision_module.md`](docs/mempatch_revision_module.md)

## MemPatch Revision Module (summary)

```text
V  ← BuildScenarioRevisionView(scenario, memories)     # Scenario View Builder
r_raw ← RevisionResponsePolicy(V)                       # Revision Response Policy
T  ← DPAConsistentProjection(authorize, parse(r_raw))   # DPA-Consistent Projection
r_final ← ProjectToBenchmarkResponse(T, r_raw)          # decision, memory_state, ...
```

Internal roles (not separate contributions): Scenario View Builder → Revision Response Policy → DPA-Consistent Projection → Benchmark-grounded Feedback.

## Paper-facing response

```json
{
  "decision": "use_current_memory",
  "memory_state": {"m1": "current", "m2": "outdated", "m3": "blocked"},
  "evidence_event_ids": ["e2", "e5"],
  "failure_diagnosis": "stale_memory_reuse",
  "answer": "..."
}
```

Vocabulary: `scenario`, `public_input`, `event_trace`, `hidden_gold`, `response`, `decision`, `memory_state`, `evidence_event_ids`, `failure_diagnosis`, `failure_mode`, `memory_status`.

## Repository layout

| Path | Role in MemPatch Revision Module |
|------|----------------------------------|
| `benchmark/retrace_bench/` | MemPatch-Bench evaluator (scores `response` vs `hidden_gold`) |
| `src/retrace_learn/` | View Builder, Response Policy, feedback |
| `src/retracemem/` | DPA-Consistent Projection (`authorize`, RevisionGate) |
| `scripts/` | Evaluate, validate, run models |
| `hf_release/mempatch_v1_1/` | Hugging Face release metadata |

Data: Hugging Face `Sylvan-Vale-Moon/MemPatch`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Evaluate predictions

```bash
PYTHONPATH=. python scripts/evaluate_retrace_bench_predictions.py \
  --data path/to/scenarios.jsonl \
  --predictions path/to/predictions.jsonl \
  --print-table
```

```python
from benchmark.retrace_bench.api import load_scenarios, load_predictions, evaluate_predictions

result = evaluate_predictions(
    load_scenarios("path/to/scenarios.jsonl"),
    load_predictions("path/to/predictions.jsonl"),
    strict=True,
)
```

## Run a model locally

```bash
git clone https://github.com/yuchenzhu-research/MemPatch.git && cd MemPatch
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,llm]" && cp .env.example .env
```

```bash
python scripts/run_retrace_bench_model.py \
  --data local/MemPatch/scenarios.jsonl \
  --provider siliconflow --model deepseek-ai/DeepSeek-V3 \
  --out-predictions local/predictions/calibration.jsonl --max-cases 10 --resume

PYTHONPATH=. python scripts/evaluate_retrace_bench_predictions.py \
  --data local/MemPatch/scenarios.jsonl \
  --predictions local/predictions/calibration.jsonl --print-table
```

## Verification

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q benchmark scripts src
```

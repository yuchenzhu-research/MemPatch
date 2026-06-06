# MemPatch

**MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents**

## Overview

Rapid Memory Integration (RMI) is the ability of an LLM agent to integrate new evidence into a shared memory basis by revising which beliefs remain usable (`current`, `outdated`, `blocked`, `unresolved`, `out_of_scope`, `restored`, …) rather than blindly appending text.

MemPatch provides (1) **MemPatch-Bench**, an evaluation-only benchmark with a fixed five-field `response` interface, and (2) the **MemPatch Revision Module**, an algorithm module that produces benchmark-compatible revision responses through typed actions, deterministic DPA verification, and projection.

Algorithm details: [`docs/mempatch_revision_module.md`](docs/mempatch_revision_module.md)  
Experiment plan: [`docs/experiments_open_source_budget_plan.md`](docs/experiments_open_source_budget_plan.md)

## What is included

| Artifact | Role |
|----------|------|
| **MemPatch-Bench** | Scenarios, `hidden_gold` (scorer-only), official evaluator |
| **MemPatch Revision Module** | View builder → policy → DPA → benchmark response projection |
| **Evaluation scripts** | `evaluate_mempatch_predictions.py`, model / module runners |
| **Smoke fixtures** | `tests/fixtures/smoke_*.jsonl` (no API cost) |
| **Release metadata** | `hf_release/mempatch_v1_1/` (manifest, checksums; JSONL not in git) |

Public evaluator import: `benchmark.mempatch_bench.api`

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

To run LLM baselines (Direct Response or Revision Module `--policy prompt`):

```bash
pip install -e ".[dev,llm]"
```

Set provider API keys in the environment (for example `SILICONFLOW_API_KEY`, `OPENAI_API_KEY`) before model calls.

## Data

The public release contains two splits only:

| Split | Rows | Purpose |
|-------|-----:|---------|
| `main` | 3000 | Broad coverage |
| `hard` | 500 | L3/L4 adversarial stress |

**Public total: 3500.** There is no train split. Scenario JSONL is **not vendored in this repository**.

1. Read split policy and checksums in `hf_release/mempatch_v1_1/manifest.json`.
2. Download `main/scenarios.jsonl` and `hard/scenarios.jsonl` from the anonymous Hugging Face dataset artifact (see manifest).
3. Place files under `local/MemPatch/main/` and `local/MemPatch/hard/` (gitignored).

## Prediction format

Each prediction row:

```json
{
  "scenario_id": "case-000001",
  "response": {
    "decision": "use_current_memory",
    "memory_state": {"m1": "current", "m2": "outdated"},
    "evidence_event_ids": ["e2"],
    "failure_diagnosis": "stale_memory_reuse",
    "answer": "..."
  }
}
```

All five `response` fields are required under strict evaluation.

## Evaluate predictions

```bash
PYTHONPATH=.:src python scripts/evaluate_mempatch_predictions.py \
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
print(result["headline_metrics"])
```

**No-cost smoke** (bundled fixtures):

```bash
PYTHONPATH=.:src python scripts/evaluate_mempatch_predictions.py \
  --data tests/fixtures/smoke_scenarios.jsonl \
  --predictions tests/fixtures/smoke_predictions.jsonl \
  --print-table
```

## Run baselines

### Direct Response

Model writes the five-field `response` directly (no DPA projection path):

```bash
PYTHONPATH=.:src python scripts/run_mempatch_model.py \
  --data local/MemPatch/main/scenarios.jsonl \
  --provider siliconflow \
  --model <OPEN_MODEL_NAME> \
  --out-predictions local/predictions/direct_main20.jsonl \
  --max-cases 20 \
  --resume
```

### MemPatch Revision Module

For reported method results, use an LLM policy (not the default smoke `noop` policy):

```bash
PYTHONPATH=.:src python scripts/run_mempatch_revision_module.py \
  --data local/MemPatch/main/scenarios.jsonl \
  --out-predictions local/predictions/module_main20.jsonl \
  --max-cases 20 \
  --policy prompt \
  --provider siliconflow \
  --model <OPEN_MODEL_NAME> \
  --resume
```

Smoke / CI only: `--policy noop` or `--policy scripted` (deterministic, no API).

### Prompt-size smoke (no API call)

```bash
PYTHONPATH=.:src python scripts/run_mempatch_model.py \
  --data tests/fixtures/smoke_scenarios.jsonl \
  --provider openai \
  --model gpt-4o-mini \
  --out-predictions local/predictions/smoke_direct.jsonl \
  --max-cases 1 \
  --print-prompt-stats

PYTHONPATH=.:src python scripts/run_mempatch_revision_module.py \
  --data tests/fixtures/smoke_scenarios.jsonl \
  --out-predictions local/predictions/smoke_module.jsonl \
  --max-cases 1 \
  --policy prompt \
  --provider openai \
  --model gpt-4o-mini \
  --print-prompt-stats
```

## Reproduce smoke tests

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile PYTHONPATH=.:src .venv/bin/python -m compileall -q benchmark scripts src tests
env PYTHONPYCACHEPREFIX=.pycache_compile PYTHONPATH=.:src .venv/bin/python -m pytest -q
```

## Expected metrics

Headline metrics (see `benchmark.mempatch_bench.api.HEADLINE_METRICS`) include:

`decision_macro_f1`, `memory_state_accuracy`, `evidence_f1`, `failure_diagnosis_accuracy`, `joint_revision_success`, `structural_revision_success`, `stale_reuse_rate`, and related auxiliary signals.

This repository ships **evaluator code and smoke fixtures only**; numeric benchmark results are produced by running the scripts above on downloaded data.

## Repository layout

```text
benchmark/mempatch_bench/   Official evaluator and model runner
src/retrace_learn/          MemPatch Revision Module runtime
src/retracemem/             Deterministic DPA kernel (internal)
scripts/                    CLI entrypoints
tests/fixtures/             Smoke scenarios and predictions
hf_release/mempatch_v1_1/   Public release metadata (no JSONL in git)
docs/                       Algorithm spec and experiment plan
```

## Limitations

- **Evaluation-only data.** Do not train on MemPatch-Bench rows intended for evaluation.
- **No learned policy weights in-repo.** Reported module runs should use `--policy prompt` (API) or `--policy scripted` (oracle smoke). A trained LoRA policy is future work.
- **Benchmark-grounded feedback is an interface only.** `reward.py` defines a decomposable reward; no training loop is wired in this artifact.
- **Scenario View Builder uses public-text heuristics** for replacement / condition / dependency extraction; it is not a gold oracle replay.
- **Projection heuristics** map DPA statuses plus public cues (for example memory id suffix `-distractor`, `Condition rule:` text) to eight-label `memory_state`; some gold labels may still require policy-side prediction or raw-response overrides.
- **JSONL must be downloaded**; a clean clone runs tests and fixture smoke but not full-bench numbers without external data.

## Citation (anonymous placeholder)

```bibtex
@misc{mempatch2026,
  title  = {MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents},
  author = {Anonymous},
  year   = {2026},
  note   = {Evaluation-only benchmark release.}
}
```

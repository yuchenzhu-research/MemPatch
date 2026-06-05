# MemPatch

**MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents**

RMI (Rapid Memory Integration) is the core problem: when an LLM agent receives new evidence, it must quickly determine whether each prior memory is still `current`, `outdated`, `blocked`, `unresolved`, `out_of_scope`, or `restored` — and produce the correct current memory state. The agent should not merely append new information.

MemPatch is **one unified paper**, not a benchmark paper plus a separate method paper:

- **MemPatch-Bench** defines scenarios, tasks, gold labels, and the evaluation interface.
- **MemPatch scaffold** learns to produce benchmark-compatible revision responses.
- **DPA** (`authorize(...)`) is the scaffold's deterministic authorization kernel: the model proposes; DPA authorizes; the benchmark evaluates the resulting memory state.
- **Benchmark-grounded feedback** turns `memory_state`, `evidence_event_ids`, and diagnostic metrics into training signal for policy improvement.

## Paper-facing response interface

The external interface readers see is a benchmark `response`:

```json
{
  "decision": "use_current_memory",
  "memory_state": {
    "m1": "current",
    "m2": "outdated",
    "m3": "blocked"
  },
  "evidence_event_ids": ["e2", "e5"],
  "failure_diagnosis": "stale_memory_reuse",
  "answer": "..."
}
```

Benchmark vocabulary: `scenario`, `public_input`, `event_trace`, `hidden_gold`, `response`, `decision`, `memory_state`, `evidence_event_ids`, `failure_diagnosis`, `failure_mode`, `memory_status`.

Inside the scaffold, three implementation roles support this interface (not three paper contributions):

1. **Scenario View Builder** — `scenario` / `event_trace` → structured revision view
2. **Revision Response Policy** — revision view → benchmark-compatible response
3. **Benchmark-grounded feedback** — response metrics → training signal

## Repository layout

| Path | Role |
|------|------|
| `benchmark/retrace_bench/` | Evaluator API, taxonomy, scorers, validation |
| `hf_release/retrace_bench_v1_1/` | Hugging Face release metadata |
| `src/retrace_learn/` | Scenario View Builder, Revision Response Policy, feedback |
| `src/retracemem/` | DPA, `authorize(...)`, memory store, multi-agent commit |
| `scripts/` | Evaluate, validate, and run models on scenarios |

Public scenario data: Hugging Face `Sylvan-Vale-Moon/ReTrace-Bench`. The repo does not track generated reports, paper drafts, run dumps, or local data copies.

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

scenarios = load_scenarios("path/to/scenarios.jsonl")
predictions = load_predictions("path/to/predictions.jsonl")
result = evaluate_predictions(scenarios, predictions, strict=True)
```

## Run a model locally

```bash
git clone https://github.com/yuchenzhu-research/MemPatch.git
cd MemPatch
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,llm]"
cp .env.example .env   # fill in provider API key
```

Download scenarios from Hugging Face into `local/ReTrace-Bench/`, then:

```bash
python scripts/run_retrace_bench_model.py \
  --data local/ReTrace-Bench/calibration/scenarios.jsonl \
  --provider siliconflow \
  --model deepseek-ai/DeepSeek-V3 \
  --out-predictions local/predictions/siliconflow_calibration.jsonl \
  --max-cases 10 \
  --resume

PYTHONPATH=. python scripts/evaluate_retrace_bench_predictions.py \
  --data local/ReTrace-Bench/calibration/scenarios.jsonl \
  --predictions local/predictions/siliconflow_calibration.jsonl \
  --out-metrics local/results/siliconflow_calibration.metrics.json \
  --print-table
```

## Verification

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q benchmark scripts src
```

# ReTrace

ReTrace contains two active code surfaces:

1. **ReTrace-Bench**: evaluation-only benchmark API, schemas, scoring, validation, taxonomy, and minimal CLI support under `benchmark/retrace_bench/`.
2. **ReTrace-Learn**: active method/runtime code under `src/retrace_learn/` and `src/retracemem/`, centered on typed revision proposals committed through deterministic `authorize(...)`.

Benchmark data is distributed through Hugging Face:
`Sylvan-Vale-Moon/ReTrace-Bench`.

The local repository intentionally does not track generated reports, paper drafts, sample files, run dumps, or local benchmark-data copies.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## ReTrace-Bench Evaluator

```bash
PYTHONPATH=. python scripts/evaluate_retrace_bench_predictions.py \
  --data path/to/scenarios.jsonl \
  --predictions path/to/predictions.jsonl \
  --print-table
```

Python API:

```python
from benchmark.retrace_bench.api import load_scenarios, load_predictions, evaluate_predictions

scenarios = load_scenarios("path/to/scenarios.jsonl")
predictions = load_predictions("path/to/predictions.jsonl")
result = evaluate_predictions(scenarios, predictions, strict=True)
```

## Run a Model and Evaluate Locally

```bash
git clone https://github.com/yuchenzhu-research/ReTrace.git
cd ReTrace
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,llm]"
```

Download the ReTrace-Bench dataset from Hugging Face:
`Sylvan-Vale-Moon/ReTrace-Bench`.

Create local API-key configuration:

```bash
cp .env.example .env
```

Fill in the provider key and run a model:

```bash
python scripts/run_retrace_bench_model.py \
  --data local/ReTrace-Bench/calibration/scenarios.jsonl \
  --provider siliconflow \
  --model deepseek-ai/DeepSeek-V3 \
  --out-predictions local/predictions/siliconflow_calibration.jsonl \
  --max-cases 10 \
  --resume
```

Score the generated predictions with the official model-agnostic evaluator:

```bash
PYTHONPATH=. python scripts/evaluate_retrace_bench_predictions.py \
  --data local/ReTrace-Bench/calibration/scenarios.jsonl \
  --predictions local/predictions/siliconflow_calibration.jsonl \
  --out-metrics local/results/siliconflow_calibration.metrics.json \
  --out-scored local/results/siliconflow_calibration.scored.jsonl \
  --print-table
```

## Verification

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q benchmark scripts src tests
.venv/bin/python -m pytest -q
```

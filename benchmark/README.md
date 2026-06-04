# ReTrace-Bench

ReTrace-Bench is an independent benchmark for **agent memory revision
reliability**: given an evolving trace of evidence and an existing memory state,
can a system correctly decide what to do, reclassify each belief, ground the
decision in the right evidence, and diagnose the failure mode — without reusing
stale, out-of-scope, or policy-invalid memory?

Each scenario is scored through four structured views:

1. **`black_box_task`** — end-to-end answer using the revised memory.
2. **`memory_state_task`** — final eligibility status of every tracked belief.
3. **`evidence_retrieval_task`** — minimal evidence event IDs justifying the decision.
4. **`diagnostic_task`** — which memory-reliability failure occurred.

This directory (`benchmark/retrace_bench/`) holds the evaluation-only code:
taxonomy, schema, scorers, validators, baselines, providers, and the public
scoring API. It is the **ReTrace-Bench** track and is independent of the
ReTrace-Learn method track.

## Official splits (ReTrace-Bench v1.0)

Four paper-facing splits, published under public names `main` / `hard` /
`realistic` / `calibration` (never train / dev / validation / test):

| HF split | On-disk path | Size | Role |
| --- | --- | --- | --- |
| `main` | `data/retrace_bench/main_3000_en/` | 3000 | Controlled benchmark main split; primary headline results. |
| `hard` | `data/retrace_bench/hard_300_en/` | 300 | Long-context / multi-evidence / multi-memory stress split. |
| `realistic` | `data/retrace_bench/realistic_100_en/` | 100 | Realistic-style workflow split. **`annotation_status = pending`**; gold not yet annotated. |
| `calibration` | `data/retrace_bench/calibration_80_en/` | 80 | Smoke / quickstart only. **Not for model selection or headline claims.** |

ReTrace-Learn (the method track) uses ReTrace-Bench-derived scenario data with
declared split roles under `data/retrace_learn/`. The benchmark track remains
method-neutral as an evaluation artifact; split roles must be explicit, and
leakage-free held-out evaluation is not claimed where the same gold labels are
used for training.
The legacy pre-v1.0 layout is recoverable from the Git tag
`legacy-retrace-bench-pre-v1.0`.

## Install / setup

The method package (`retracemem`) is pip-installable. The evaluation-only
`benchmark.retrace_bench` package is **not** installed by pip — import it via
`PYTHONPATH=.` from the repo root.

```bash
pip install -e ".[dev]"
export PYTHONPATH=.      # required for `import benchmark.retrace_bench...`
```

Every benchmark command below is therefore prefixed with `PYTHONPATH=.`.

## Load the dataset

From local JSONL (source of truth):

```python
from benchmark.retrace_bench.api import load_scenarios

scenarios = load_scenarios("data/retrace_bench/main_3000_en")
# or point directly at a scenarios.jsonl file
```

From Hugging Face (`Sylvan-Vale-Moon/ReTrace-Bench`):

```python
import json
from datasets import load_dataset

ds = load_dataset("Sylvan-Vale-Moon/ReTrace-Bench")  # main / hard / realistic / calibration
row = ds["main"][0]
hidden_gold = json.loads(row["hidden_gold_json"])  # nested fields are JSON string columns
```

The HF viewer publishes nested structures as JSON string columns
(`public_input_json`, `tasks_json`, `hidden_gold_json`, `metadata_json`,
`secondary_failure_modes_json`); parse them with `json.loads(...)`. Benchmark
rows carry no `training_targets`. The local `data/` files keep the native nested
JSONL schema.

## Validate a split

```bash
PYTHONPATH=. python scripts/validate_retrace_bench_dataset.py \
  --data data/retrace_bench/main_3000_en/scenarios.jsonl
```

## Run built-in baselines

Offline baselines need no API keys (`latest_only`, `retrieve_all`,
`rag_lexical`, `crud_memory`, `mem0_style`, `heuristic_memory_state`, and the
`retrace_oracle_engine` gold-replay reference):

```bash
PYTHONPATH=. python scripts/run_retrace_bench_baseline.py \
  --data data/retrace_bench/main_3000_en/scenarios.jsonl \
  --baseline latest_only \
  --out outputs/retrace_bench/latest_only.jsonl
```

The full offline matrix:

```bash
PYTHONPATH=. python scripts/run_retrace_bench_ablation.py \
  --data data/retrace_bench/main_3000_en/scenarios.jsonl \
  --out-dir outputs/retrace_bench/ablation \
  --resume
```

For live API baselines, always keep `--resume` on long runs. The runner reads
the existing prediction JSONL, skips completed `scenario_id`s, and appends only
the remaining cases.

## Evaluate external predictions

The official evaluator runs no model and needs no API keys — it only scores a
JSONL predictions file against a split:

```bash
PYTHONPATH=. python scripts/evaluate_retrace_bench_predictions.py \
  --data data/retrace_bench/main_3000_en/scenarios.jsonl \
  --predictions path/to/predictions.jsonl \
  --out-metrics outputs/retrace_bench/my_model.metrics.json \
  --out-scored outputs/retrace_bench/my_model.scored.jsonl \
  --print-table
```

See `examples/retrace_bench/` for a runnable example and the full prediction
schema (decision / memory_state / evidence_event_ids / failure_diagnosis /
answer).

## Python API

```python
from benchmark.retrace_bench.api import (
    load_scenarios, load_predictions, evaluate_predictions,
    HEADLINE_METRICS, AUXILIARY_METRICS,
)

scenarios = load_scenarios("data/retrace_bench/main_3000_en")
predictions = load_predictions("my_model.predictions.jsonl")
result = evaluate_predictions(scenarios, predictions, strict=True)

print(result["count"], result["headline_metrics"])
# strict=False instead collects result["warnings"] / result["errors"]
# and scores whatever is valid.
```

## Headline metrics

Paper-facing headline metrics (`HEADLINE_METRICS`):

- `decision_macro_f1` — primary decision metric (robust to the dominant
  `use_current_memory` class).
- `non_answer_decision_accuracy`
- `memory_state_accuracy`
- `evidence_f1`
- `failure_diagnosis_accuracy`
- `stale_reuse_rate`

`AUXILIARY_METRICS` (diagnostic, not headline) include
`black_box_decision_accuracy`, which can be dominated by the majority decision
class and should not be read as the primary decision score.

## Oracle boundary

`retrace_oracle_engine` replays the hidden gold decision/state/evidence through
the deterministic ReTrace-Engine. It is a mechanism/upper-bound **reference**,
not a deployable black-box baseline, and must not be compared against learned or
prompt-based systems as if it were one.

## More

- Benchmark paper draft: [`docs/retrace_bench/benchmark_paper.md`](../docs/retrace_bench/benchmark_paper.md)
- Manual validation protocol:
  [`manual_validation_protocol.md`](../docs/retrace_bench/manual_validation_protocol.md).
  Legacy pre-v1 artifacts were removed from the active tree; current results must
  be regenerated on v1.0 splits.
- Hugging Face dataset card: [`release/huggingface/ReTrace-Bench/README.md`](../release/huggingface/ReTrace-Bench/README.md)
- Benchmark-paper workspace: [`papers/retrace_bench/`](../papers/retrace_bench/README.md)

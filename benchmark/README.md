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

## Official splits

| HF split | On-disk path | Role |
| --- | --- | --- |
| `test` | `data/retrace_bench/test_800_templateheldout_en/` | Canonical, paper-facing held-out benchmark. **Do not train/tune/select on it.** |
| `validation` | `data/retrace_bench/sample_80_hard_en/` | Calibration/quickstart (80 hard scenarios). HF `validation` is **viewer compatibility only** — not model/checkpoint selection. |
| `train` | `data/retrace_supervision/train_3000_en/` | Supervision pool for learning-based proposers; may contain `training_targets`. Not a test. |
| `dev` | `data/retrace_supervision/dev_400_en/` | Selection pool. Not a test. |

The old `data/retrace_bench/test_800_en/` is a prototype/diagnostic split and is
excluded from the public release.

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

scenarios = load_scenarios("data/retrace_bench/test_800_templateheldout_en")
# or point directly at a scenarios.jsonl file
```

From Hugging Face (`Sylvan-Vale-Moon/ReTrace-Bench`):

```python
import json
from datasets import load_dataset

ds = load_dataset("Sylvan-Vale-Moon/ReTrace-Bench")  # test / validation / train / dev
row = ds["test"][0]
hidden_gold = json.loads(row["hidden_gold_json"])  # nested fields are JSON string columns
```

The HF viewer publishes nested structures as JSON string columns
(`public_input_json`, `tasks_json`, `hidden_gold_json`, `metadata_json`,
`secondary_failure_modes_json`, `training_targets_json`); parse them with
`json.loads(...)`. The local `data/` files keep the native nested JSONL schema.

## Validate a split

```bash
PYTHONPATH=. python scripts/validate_retrace_bench_dataset.py \
  --data data/retrace_bench/test_800_templateheldout_en/scenarios.jsonl
```

## Run built-in baselines

Offline baselines need no API keys (`latest_only`, `retrieve_all`,
`rag_lexical`, `crud_memory`, `mem0_style`, `heuristic_memory_state`, and the
`retrace_oracle_engine` gold-replay reference):

```bash
PYTHONPATH=. python scripts/run_retrace_bench_baseline.py \
  --data data/retrace_bench/test_800_templateheldout_en/scenarios.jsonl \
  --baseline latest_only \
  --out outputs/retrace_bench/latest_only.jsonl
```

The full offline matrix:

```bash
PYTHONPATH=. python scripts/run_retrace_bench_ablation.py \
  --data data/retrace_bench/test_800_templateheldout_en/scenarios.jsonl \
  --out-dir outputs/retrace_bench/ablation
```

## Evaluate external predictions

The official evaluator runs no model and needs no API keys — it only scores a
JSONL predictions file against a split:

```bash
PYTHONPATH=. python scripts/evaluate_retrace_bench_predictions.py \
  --data data/retrace_bench/test_800_templateheldout_en/scenarios.jsonl \
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

scenarios = load_scenarios("data/retrace_bench/test_800_templateheldout_en")
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
- Baselines on the canonical split: [`docs/retrace_bench/baseline_results_test_800_templateheldout_en.md`](../docs/retrace_bench/baseline_results_test_800_templateheldout_en.md)
- Manual validation protocol: [`docs/retrace_bench/manual_validation_protocol.md`](../docs/retrace_bench/manual_validation_protocol.md)
- Hugging Face dataset card: [`release/huggingface/ReTrace-Bench/README.md`](../release/huggingface/ReTrace-Bench/README.md)
- Benchmark-paper workspace (blind-review safe): [`papers/retrace_bench/`](../papers/retrace_bench/README.md)
- Submission readiness checklist: [`docs/retrace_bench/submission_readiness.md`](../docs/retrace_bench/submission_readiness.md)

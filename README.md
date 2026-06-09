# MemPatch

**MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents**

RMI is the ability of an LLM agent to revise which beliefs remain usable (`current`, `outdated`, `blocked`, `unresolved`, …) when new evidence arrives — not blindly append text.

This repo ships:

| Component | Location |
|-----------|----------|
| **MemPatch-Bench** (dataset view + scoring) | `benchmark/` |
| **Scenario generator** (v1.3) | `benchmark/generation/` |
| **Revision module** (Path A/B: view → proposer → projection) | `src/mempatch_learn/` |
| **DPA kernel** (deterministic verifier; not the scorer) | `src/mempatch_dpa/` |
| **Dataset release** | `hf_release/mempatch/` |

## Architecture

Responsibilities are split so the benchmark never implements the revision method, and the method never defines metrics.

```text
benchmark/                    dataset format, public view, scoring ONLY
  api.py                      evaluate_predictions() — the single public scorer
  public_view.py              model-visible scenario fields (no hidden_gold)
  model_runner.py             optional JSON prompt adapter for baselines (NOT revision)

src/mempatch_learn/           Path A/B method
  runtime/scenario_revision.py   build revision view V from scenario + memory
  runtime/learned_proposer.py      proposer πθ(V) → typed actions or JSON
  runtime/dpa_runtime.py         DPA-consistent projection
  runtime/benchmark_projection.py  project to five-field benchmark response

src/mempatch_dpa/             DPA authorization kernel (authorize)
                              used by the revision module, not by the scorer

scripts/                      thin CLIs that wire runners to the evaluator
  eval/                       produce predictions.jsonl
  workflows/                  audit, validate, score, paper pipeline
  data/                       generate, package, bundle datasets
  mlx_support/                        MLX download + chat helpers
  memory/                     RAG / full / mem0 context builders
  analysis/                   optional error breakdowns
```

**No overlap:** `benchmark/` scores any compliant `predictions.jsonl`. `src/` implements how Path A/B (and DPA) produce those predictions. External baselines use `benchmark.model_runner.build_prompt` as a JSON port, not the revision stack.

## Data flow

Every experiment line follows the same contract:

```text
hf_release/mempatch/{split}/scenarios.jsonl
        │
        ▼
scripts/eval/run_lora_test_eval.py  →  scripts/workflows/evaluate_mempatch_predictions.py
        │
        ▼
predictions.jsonl              { "scenario_id", "response": { five fields } }
        │
        ▼
benchmark.api.evaluate_predictions()
        │
        ▼
metrics JSON / headline table
```

Scoring always goes through `benchmark.api` (CLI: `scripts/workflows/evaluate_mempatch_predictions.py`). Training: `scripts/workflows/run_kfold_train.sh`. See `scripts/README.md`.

Headline metrics: `joint_revision_success`, `decision_macro_f1`, `memory_state_accuracy`, `evidence_f1`, `failure_diagnosis_accuracy`, `stale_reuse_rate`, `response_schema_compliance_rate`.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

LLM baselines: `pip install -e ".[dev,llm]"`

## Dataset (v1.3)

| Split | Rows | Use |
|-------|-----:|-----|
| `train` | 3500 | SFT; stratified 5-fold for checkpoint selection |
| `test` | 500 | Held-out final eval |

v1.3 uses 7 primary failure modes, 8 pattern families, and 6 domains. Difficulty: train `L3`; test `L4`.

Regenerate:

```bash
PYTHONPATH=.:src python scripts/data/generate_mempatch.py --full --out-dir hf_release/mempatch
```

Audit gate (must pass before training):

```bash
PYTHONPATH=.:src python scripts/workflows/audit_decision_boundary.py \
  --data hf_release/mempatch/train \
  --data hf_release/mempatch/test
```

## Evaluate predictions

```bash
PYTHONPATH=.:src python scripts/workflows/evaluate_mempatch_predictions.py \
  --data hf_release/mempatch/test/scenarios.jsonl \
  --predictions path/to/predictions.jsonl \
  --print-table
```

```python
from benchmark.api import load_scenarios, load_predictions, evaluate_predictions

result = evaluate_predictions(
    load_scenarios("hf_release/mempatch/test/scenarios.jsonl"),
    load_predictions("predictions.jsonl"),
    strict=True,
)
print(result["headline_metrics"])
```

Smoke (no API):

```bash
PYTHONPATH=.:src python scripts/workflows/evaluate_mempatch_predictions.py \
  --data tests/fixtures/smoke_scenarios.jsonl \
  --predictions tests/fixtures/smoke_predictions.jsonl
```

## Train + eval (Path B LoRA)

```bash
RUN_ID=full256 KFOLD_FOLD=0 bash scripts/workflows/run_kfold_train.sh qwen3_14b
ADAPTER=local/adapters/qwen3_14b_pathB_lora/fold0/full256 \
  bash scripts/workflows/run_eval_test.sh
```

Download MLX base weights into `local/models/` via [MLX Community on Hugging Face](https://huggingface.co/mlx-community) (e.g. `Qwen3-14B-MLX-4bit`).

## Local workspace (`local/`, gitignored)

Keep generated artifacts out of git. Recommended layout:

```text
local/
  models/           MLX base weights (e.g. Qwen3-14B-MLX-4bit)
  adapters/         LoRA checkpoints per backbone
  data/             SFT bundles, smoke slices, eval subsets
  runs/             eval outputs
  logs/             training and pipeline logs
```

Do not `git add local/`. See `AGENTS.md` for agent workflow and pre-commit cache cleanup.

## Repository layout

```text
benchmark/              MemPatch-Bench: public view, taxonomy, evaluate_predictions
src/mempatch_learn/     Revision module (view → proposer → DPA → projection)
src/mempatch_dpa/       DPA authorization kernel (deterministic, not scoring)
scripts/
  eval/                 prediction runners
  workflows/            audit, validate, score, paper pipeline
  data/                 generate, package, bundle
  mlx_support/                  MLX utilities
  memory/               external memory baseline helpers
  analysis/             optional diagnostics
tests/                  unit tests + smoke fixtures
hf_release/mempatch/    dataset bundle (4000 scenarios)
config/                 paper model cards (params, colors)
data/mempatch/          tracked audit artifacts
```

Script index: `scripts/README.md`.

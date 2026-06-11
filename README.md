# MemPatch

**MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents**

RMI is the ability of an LLM agent to revise which beliefs remain usable (`current`, `outdated`, `blocked`, `unresolved`, …) when new evidence arrives — not blindly append text.

| Component | Location |
|-----------|----------|
| **MemPatch-Bench** (dataset view + scoring) | `benchmark/` |
| **Scenario generator** (v1.3) | `benchmark/generation/` |
| **Revision module** (Path A/B) | `mempatch/revision/` (`import mempatch.revision`) |
| **DPA kernel** (deterministic verifier) | `mempatch/dpa/` (`import mempatch.dpa`) |
| **Regression tests** | `mempatch/tests/` |
| **Reproducibility CLIs** | `scripts/` |

The scenario JSONL bundle is **not** checked into this repository. Regenerate it locally (see [Dataset](#dataset-v13)) or load from a public dataset host after release.

## Architecture

```text
benchmark/                    dataset format, public view, scoring ONLY
  api.py                      evaluate_predictions() — the single public scorer
  public_view.py              model-visible scenario fields (no hidden_gold)
  model_runner.py             optional JSON prompt adapter for baselines (NOT revision)

mempatch/revision/           Path A/B method (view → proposer → DPA → projection)
mempatch/dpa/                DPA authorization kernel (authorize)
scripts/                      thin CLIs that wire runners to the evaluator
```

**No overlap:** `benchmark/` scores any compliant `predictions.jsonl`. `mempatch/` implements how Path A/B produce those predictions. External baselines use `benchmark.model_runner.build_prompt` as a JSON port, not the revision stack.

## Data flow

```text
{split}/scenarios.jsonl          # local or downloaded bundle
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

Paper metric hierarchy:

| Tier | Metrics | Interpretation |
|------|---------|----------------|
| Primary effects | `decision_macro_f1`, `memory_state_accuracy` | Balanced decision quality and memory revision correctness |
| Primary validity gate | `response_schema_compliance_rate` | Operational validity; report both raw and projected rates |
| Secondary | `evidence_f1`, `failure_diagnosis_accuracy` | Evidence selection and diagnostic quality |
| Confirmatory strict composite | `joint_revision_success` | Exact all-or-nothing success; expected to be sparse on L4 |
| Safety / diagnostics | `stale_reuse_rate`, answer and per-mode metrics | Failure analysis rather than the main efficacy claim |

The strict joint metric is not relaxed after evaluation. Sparse or zero values
are reported alongside component metrics instead of changing the scorer.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Optional LLM baselines: `pip install -e ".[dev,llm]"` (export provider API keys yourself).

## Dataset (v1.3)

| Split | Rows | Use |
|-------|-----:|-----|
| `train` | 3500 | SFT; one fixed stratified 80/20 train/validation split |
| `test` | 500 | Held-out final eval |

v1.3 uses 7 primary failure modes, 8 pattern families, and 6 domains. Difficulty: train `L3`; test `L4`.

Regenerate locally (default output directory is gitignored):

```bash
python scripts/data/generate_mempatch.py --full --out-dir local/data/mempatch
python scripts/data/package_mempatch_release.py \
  --input-dir local/data/mempatch --out-dir local/data/mempatch --validate
```

Audit gate (must pass before training):

```bash
python scripts/workflows/audit_decision_boundary.py \
  --data local/data/mempatch/train \
  --data local/data/mempatch/test
```

## Evaluate predictions

```bash
python scripts/workflows/evaluate_mempatch_predictions.py \
  --data local/data/mempatch/test/scenarios.jsonl \
  --predictions path/to/predictions.jsonl \
  --print-table
```

```python
from benchmark.api import load_scenarios, load_predictions, evaluate_predictions

result = evaluate_predictions(
    load_scenarios("local/data/mempatch/test/scenarios.jsonl"),
    load_predictions("predictions.jsonl"),
    strict=True,
)
print(result["headline_metrics"])
```

Smoke (no dataset download; uses committed fixtures):

```bash
python scripts/workflows/evaluate_mempatch_predictions.py \
  --data mempatch/tests/fixtures/smoke_scenarios.jsonl \
  --predictions mempatch/tests/fixtures/smoke_predictions.jsonl
```

## Train + eval (Path B ablation, MLX)

```bash
RUN_ID=full256 KFOLD_FOLD=0 bash scripts/workflows/run_kfold_train.sh qwen3_14b
ADAPTER=local/adapters/qwen3_14b_pathB_lora/fold0/full256 \
  bash scripts/workflows/run_eval_test.sh
```

Download MLX base weights into `local/models/` via [MLX Community on Hugging Face](https://huggingface.co/mlx-community).

## Local workspace (`local/`, gitignored)

```text
local/
  data/mempatch/    generated scenario JSONL (train + test)
  models/           MLX base weights
  adapters/         LoRA checkpoints
  runs/             eval outputs
  logs/             training logs
```

## Repository layout

```text
benchmark/              MemPatch-Bench scorer + generator
mempatch/
  revision/             Path A/B revision module
  dpa/                  Defeat-Path Authorization kernel
  tests/                pytest + smoke fixtures
scripts/                reproducibility CLIs (eval, workflows, data)
data/mempatch/          tracked boundary-audit artifact (v1.3)
```

Script index: `scripts/README.md`.

## Reproduction models (local LoRA)

| Slug | Model | Params |
|------|-------|-------:|
| `qwen3_14b` | Qwen3-14B | 14B |
| `gemma3_12b` | Gemma 3 12B Instruct | 12B |
| `mistral_nemo_12b` | Mistral Nemo 12B Instruct | 12B |
| `llama3_1_8b` | Llama 3.1 8B Instruct | 8B |

All local runs use 4-bit MLX weights and temperature 0. Full hyperparameters belong in the paper appendix.

## Naming (paper ↔ code)

| Paper term | Code import / path |
|------------|-------------------|
| **MemPatch** | Project name; PyPI package `mempatch` |
| **MemPatch-Bench** | `benchmark/` — dataset view + `evaluate_predictions` |
| **MemPatch revision module** | `mempatch.revision` — view → proposer → DPA → projection |
| **DPA** (Defeat-Path Authorization) | `mempatch.dpa.authorize` |
| **Path A (full MemPatch)** | Revision view → typed action proposer → DPA → benchmark projection |
| **Path B (ablation)** | Direct five-field response JSON without the typed revision/DPA stack |

The current Linux `test500_base` and `test500_lora_best` runs are Path B
direct-response evaluations. The typed proposer and DPA runtime exist in
`mempatch/revision/`, but are not yet wired into the Linux evaluator.

## Citation

If you use MemPatch-Bench or this code, please cite the accompanying paper (bibtex to be added upon publication).

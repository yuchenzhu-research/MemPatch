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
scripts/eval/* runner          (or scripts/workflows/evaluate_mempatch_predictions.py on existing files)
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

Runners live under `scripts/eval/`; scoring always goes through `benchmark.api` (CLI wrapper: `scripts/workflows/evaluate_mempatch_predictions.py`).

## Experiment lines

External memory baselines (RAG, full-context, mem0, base) and DirectJudge all call `build_prompt()` from `benchmark.model_runner`. The prompt embeds `required_output_schema` with the five response fields and valid enum strings — the model must return **strict JSON**, not free text. See `scripts/memory/mempatch_memory_context.py` for per-backend context filtering before `build_prompt`.

| Line | Script | Mechanism | Trained? |
|------|--------|-----------|----------|
| **External memory** (`base`, `full`, `rag`, `mem0`) | `scripts/eval/run_mempatch_memory_baselines.py` | filtered public view → `build_prompt` → MLX → five-field JSON | No |
| **DirectJudge** | `scripts/eval/run_mempatch_model.py` | API/MLX provider → same five-field JSON schema | Optional |
| **Path A** (base projection) | `scripts/eval/run_mlx_revision_module_eval.py` | revision view → typed actions JSON → DPA → projection → five fields | No (base MLX) |
| **Path B** (adapted) | `scripts/eval/run_mlx_lora_smoke_eval.py` | SFT prompt → MLX+LoRA → direct five-field JSON | Yes (LoRA) |
| **Revision module (dev)** | `scripts/eval/run_mempatch_revision_module.py` | scripted or prompt policy; same DPA → projection stack | Optional |

Fair comparisons: RAG/full/mem0 vs Path B (same JSON protocol); Path A vs Path B (same stack, training differs).

Headline metric: `joint_revision_success` (decision + memory_state + evidence F1=1 + answer consistency + no stale reuse). Full metric definitions: `benchmark/scorers_general.py`, `scripts/README.md`.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

LLM baselines: `pip install -e ".[dev,llm]"`

## Dataset (v1.3)

| Split | Rows | Use |
|-------|-----:|-----|
| `train` | 2700 | Fine-tuning only |
| `validation` | 800 | Dev eval |
| `test` | 500 | Held-out final eval |

v1.3 uses 7 primary failure modes, 8 pattern families, and 6 domains. Difficulty labels: train/validation `L3`; test `L4`.

Regenerate:

```bash
PYTHONPATH=.:src python scripts/data/generate_mempatch.py --full --out-dir hf_release/mempatch
```

Audit gate (must pass before training):

```bash
PYTHONPATH=.:src python scripts/workflows/audit_decision_boundary.py \
  --data hf_release/mempatch/train \
  --data hf_release/mempatch/validation \
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

## Revision module smoke

```bash
PYTHONPATH=.:src python scripts/eval/run_mempatch_revision_module.py \
  --data tests/fixtures/smoke_scenarios.jsonl \
  --out-predictions /tmp/revision_smoke.jsonl \
  --max-cases 1 --policy scripted \
  --scripted-actions tests/fixtures/scripted_noop_actions.json
```

## MLX paper experiments (Qwen3 / Gemma 3 / Mistral Nemo / Llama 3.1)

Default matrix: four open models at 8B–14B with identical `paper` LoRA (256 iters):

```bash
bash scripts/workflows/run_paper_pipeline.sh
SKIP_DOWNLOAD=1 bash scripts/workflows/run_paper_pipeline.sh
```

Download weights only:

```bash
.venv/bin/python scripts/mlx_support/download_mlx_model.py --preset mistral-nemo-12b --mirror --disable-xet
.venv/bin/python scripts/mlx_support/download_mlx_model.py --preset llama-3.1-8b-instruct --mirror --disable-xet
```

## Local workspace (`local/`, gitignored)

Keep generated artifacts out of git. Recommended layout:

```text
local/
  models/           MLX base weights (e.g. Qwen3-14B-MLX-4bit)
  adapters/         LoRA checkpoints per backbone
  data/             SFT bundles, smoke slices, eval subsets
  runs/
    paper/          paper pipeline predictions + metrics
    baselines/      RAG / full / mem0 / base runs
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

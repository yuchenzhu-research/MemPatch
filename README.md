# MemPatch

**MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents**

RMI is the ability of an LLM agent to revise which beliefs remain usable (`current`, `outdated`, `blocked`, `unresolved`, …) when new evidence arrives — not blindly append text.

This repo ships:

| Component | Location |
|-----------|----------|
| **MemPatch-Bench** (evaluator + taxonomy) | `benchmark/` |
| **Scenario generator** (v1.3) | `benchmark/generation/` |
| **Revision module** (learned proposer + projection) | `src/mempatch_learn/` |
| **DPA kernel** (deterministic verifier; not the benchmark scorer) | `src/mempatch_dpa/` |
| **Dataset release** | `hf_release/mempatch/` |

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

v1.3 uses 7 primary failure modes, 8 pattern families, and 6 domains. The
taxonomy module keeps additional reserved labels for future releases and API
compatibility, but model-facing v1.3 prompts and release validation use the
primary sets. Difficulty labels are short labels: train/validation are `L3`;
test is `L4`.

Regenerate:

```bash
PYTHONPATH=.:src python scripts/generate_mempatch.py --full --out-dir hf_release/mempatch
```

## Evaluate predictions

```bash
PYTHONPATH=.:src python scripts/evaluate_mempatch_predictions.py \
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
PYTHONPATH=.:src python scripts/evaluate_mempatch_predictions.py \
  --data tests/fixtures/smoke_scenarios.jsonl \
  --predictions tests/fixtures/smoke_predictions.jsonl
```

## Revision module smoke

```bash
PYTHONPATH=.:src python scripts/run_mempatch_revision_module.py \
  --data tests/fixtures/smoke_scenarios.jsonl \
  --out-predictions /tmp/revision_smoke.jsonl \
  --max-cases 1 --policy scripted \
  --scripted-actions tests/fixtures/scripted_noop_actions.json
```

## MLX paper experiments (Qwen3.5 / Gemma 4 / DeepSeek-R1)

```bash
# Full pipeline: download → DeepSeek smoke → LoRA (256 iters) → test500 → figures
bash scripts/run_paper_pipeline.sh

# Download only (hf-mirror by default)
bash scripts/download_paper_models.sh
```

## Layout

```
benchmark/              MemPatch-Bench: evaluator, taxonomy, scenario generator
src/mempatch_learn/     Revision module (view → proposer → projection)
src/mempatch_dpa/       DPA authorization kernel (deterministic, not scoring)
scripts/                Core CLIs + MLX paper pipeline (see scripts/run_paper_pipeline.sh)
tests/                  Unit tests + smoke fixtures
hf_release/mempatch/    Dataset bundle (4000 scenarios)
config/                 Paper model cards (params, colors)
data/mempatch/          Tracked audit artifacts
```

See `AGENTS.md` for agent workflow and pre-commit cache cleanup.

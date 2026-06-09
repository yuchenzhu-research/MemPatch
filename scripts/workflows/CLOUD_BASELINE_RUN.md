# Cloud agent: external memory baselines (4 models × RAG + Full)

## Goal

Run **100 test cases** from `hf_release/mempatch/test/scenarios.jsonl` for each combination:

| Model slug | Local MLX dir |
|------------|---------------|
| `llama3_1_8b` | `local/models/Meta-Llama-3.1-8B-Instruct-4bit` |
| `mistral_nemo_12b` | `local/models/Mistral-Nemo-Instruct-2407-4bit` |
| `gemma3_12b` | `local/models/gemma-3-12b-it-4bit` |
| `qwen3_14b` | `local/models/Qwen3-14B-MLX-4bit` |

Backends: **`rag`**, **`full`** (both use `build_prompt` with full JSON schema + enum options → MLX base → `benchmark.api` scoring).

Output: `local/runs/baselines/external_memory/{slug}_{backend}_test100_*.{jsonl,json}`

## Requirements

- **macOS arm64** (MLX does not run on Linux CI).
- Python **3.12+**, repo at MemPatch root.
- Four MLX 4-bit models under `local/models/` (download if missing).
- `scripts/mlx_support/` (not `scripts/mlx/`) — avoids shadowing PyPI `mlx`.

## Setup (once)

```bash
cd MemPatch
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e ".[mlx,dev]"

export PYTHONPATH=.:src
python -c "
from scripts._root import bootstrap_from
bootstrap_from('scripts/eval/run_mempatch_memory_baselines.py')
import mlx.core as mx
print('mlx ok', mx.__version__)
"
mkdir -p local/runs/baselines/external_memory local/models
```

## Download models (if missing)

```bash
export PYTHONPATH=.:src
for preset in llama-3.1-8b-instruct mistral-nemo-12b gemma3-12b qwen3-14b; do
  python scripts/mlx_support/download_mlx_model.py --preset "$preset" --mirror --disable-xet
done
```

## Run (recommended: one model at a time)

Each command runs **RAG + Full** for that model on **100 cases**, then prints summary.

```bash
cd MemPatch && source .venv/bin/activate
export PYTHONPATH=.:src
export LIMIT=100
export OUT=local/runs/baselines/external_memory
mkdir -p "$OUT"

MODELS=llama3_1_8b bash scripts/workflows/run_external_memory_baselines.sh
MODELS=mistral_nemo_12b bash scripts/workflows/run_external_memory_baselines.sh
MODELS=gemma3_12b bash scripts/workflows/run_external_memory_baselines.sh
MODELS=qwen3_14b bash scripts/workflows/run_external_memory_baselines.sh
```

Or all four in one job (long):

```bash
LIMIT=100 bash scripts/workflows/run_external_memory_baselines.sh
```

Force rerun: `FORCE=1 LIMIT=100 MODELS=qwen3_14b bash scripts/workflows/run_external_memory_baselines.sh`

## Compare to MemPatch Path B (optional, per model)

After baselines, run LoRA on same 100 cases (requires adapter + SFT bundle):

```bash
PYTHONPATH=.:src python scripts/eval/run_mlx_lora_smoke_eval.py \
  --data local/train_data/paper/test500/sft.jsonl \
  --eval-data hf_release/mempatch/test/scenarios.jsonl \
  --limit 100 \
  --model local/models/Qwen3-14B-MLX-4bit \
  --adapter-path local/adapters/qwen3_14b_pathB_lora \
  --out-predictions local/runs/baselines/external_memory/qwen3_14b_pathB_lora_test100_predictions.jsonl \
  --out-metrics local/runs/baselines/external_memory/qwen3_14b_pathB_lora_test100_metrics.json
```

## Success criteria

- 8 metrics files for baselines: `{slug}_{rag|full}_test100_metrics.json`
- Each has `headline_metrics.joint_revision_success`, `decision_macro_f1`, `evidence_f1`
- Log: `local/runs/baselines/external_memory/run.log`

## Architecture (for agent)

```text
scenarios.jsonl → build_baseline_view(rag|full) → build_prompt(JSON+enums) → MLX → predictions.jsonl → evaluate_predictions
```

Scoring only in `benchmark/`; method code in `src/mempatch_learn/`. Baselines do **not** use Path A/B unless explicitly run.

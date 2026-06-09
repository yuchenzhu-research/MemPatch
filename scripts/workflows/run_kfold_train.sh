#!/usr/bin/env bash
# Train Path B LoRA on one k-fold with an isolated adapter directory (no overwrite).
#
#   RUN_ID=exp001 KFOLD_FOLD=0 bash scripts/workflows/run_kfold_train.sh qwen3_14b
#   RESUME_FROM=local/adapters/qwen3_14b_pathB_lora/fold0/exp001/0000128_adapters.safetensors \
#     RUN_ID=exp001_cont256 bash scripts/workflows/run_kfold_train.sh qwen3_14b
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-$ROOT:$ROOT/src:$ROOT/scripts}"

SLUG="${1:?usage: run_kfold_train.sh <qwen3_14b|gemma3_12b|mistral_nemo_12b|llama3_1_8b>}"
PROFILE="${PROFILE:-paper}"
KFOLDS="${KFOLDS:-5}"
KFOLD_FOLD="${KFOLD_FOLD:-0}"
RUN_ID="${RUN_ID:?set RUN_ID to a unique name per training run}"
SEED="${SEED:-42}"

case "$SLUG" in
  qwen3_14b) MODEL_DIR="$ROOT/local/models/Qwen3-14B-MLX-4bit" ;;
  gemma3_12b) MODEL_DIR="$ROOT/local/models/gemma-3-12b-it-4bit" ;;
  mistral_nemo_12b) MODEL_DIR="$ROOT/local/models/Mistral-Nemo-Instruct-2407-4bit" ;;
  llama3_1_8b) MODEL_DIR="$ROOT/local/models/Meta-Llama-3.1-8B-Instruct-4bit" ;;
  *) echo "unknown slug: $SLUG" >&2; exit 1 ;;
esac

SFT_DIR="$ROOT/local/train_data/kfold/${SLUG}_fold${KFOLD_FOLD}"
ADAPTER_ROOT="$ROOT/local/adapters/${SLUG}_pathB_lora"
MLX_CONFIG="$ROOT/local/logs/kfold/${SLUG}_fold${KFOLD_FOLD}_${RUN_ID}.yaml"
mkdir -p "$ROOT/local/logs/kfold" "$SFT_DIR"

PREPARE_ARGS=(
  --profile "$PROFILE" --full-train
  --out-dir "$SFT_DIR"
  --model-dir "$MODEL_DIR"
  --adapter-dir "$ADAPTER_ROOT"
  --mlx-config "$MLX_CONFIG"
  --seed "$SEED"
  --k-folds "$KFOLDS" --fold "$KFOLD_FOLD"
  --run-id "$RUN_ID"
)
if [[ -n "${RESUME_FROM:-}" ]]; then
  PREPARE_ARGS+=(--resume-from "$RESUME_FROM")
fi

"$PYTHON" "$ROOT/scripts/data/prepare_mempatch_v13_smoke.py" "${PREPARE_ARGS[@]}"
"$PYTHON" -m mlx_lm lora --config "$MLX_CONFIG"

echo "Adapter dir: $ADAPTER_ROOT/fold${KFOLD_FOLD}/${RUN_ID}"

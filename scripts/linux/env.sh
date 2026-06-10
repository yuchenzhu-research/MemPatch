#!/usr/bin/env bash
# Shared paths for Linux CUDA paper runs. Source from other scripts/linux/*.sh
set -euo pipefail

LINUX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$LINUX_DIR/../.." && pwd)"

export ROOT
export PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-$ROOT:$ROOT/src:$ROOT/scripts}"

# Paper protocol (matches MLX profile `paper` in prepare_mempatch_v13_smoke.py)
export PROFILE="${PROFILE:-paper}"
export KFOLDS="${KFOLDS:-5}"
export SEED="${SEED:-42}"
export RUN_ID="${RUN_ID:-full384}"
export TRAIN_ITERS="${TRAIN_ITERS:-384}"
export SAVE_EVERY="${SAVE_EVERY:-64}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"
export EVAL_LIMIT="${EVAL_LIMIT:-}"

# Local artifact layout (under repo-root local/, gitignored)
export LOCAL_ROOT="${LOCAL_ROOT:-$ROOT/local}"
export TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-$LOCAL_ROOT/train_data/kfold}"
export ADAPTER_ROOT="${ADAPTER_ROOT:-$LOCAL_ROOT/adapters}"
export RESULTS_ROOT="${RESULTS_ROOT:-$LOCAL_ROOT/results}"
export LOG_ROOT="${LOG_ROOT:-$LOCAL_ROOT/logs/kfold}"
export LOCAL_MODEL_ROOT="${LOCAL_MODEL_ROOT:-$LOCAL_ROOT/models}"
export HF_CACHE="${HF_HOME:-$LOCAL_ROOT/hf_cache}"
export PIPELINE_LOG="${PIPELINE_LOG:-$LOCAL_ROOT/logs/pipeline.log}"

# AutoDL: HF mirror + disable xet CDN (hangs at "Fetching N files: 0%").
# Do NOT default to HF_HUB_OFFLINE — use prefetch_model.sh -> LOCAL_MODEL_ROOT instead.
# Set HF_ENDPOINT= to force the official Hugging Face endpoint.
export HF_ENDPOINT="${HF_ENDPOINT-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_DOWNLOAD_WORKERS="${HF_DOWNLOAD_WORKERS:-1}"

export TEST_SCENARIOS="${TEST_SCENARIOS:-$ROOT/hf_release/mempatch/test/scenarios.jsonl}"
export TEST_SFT_DIR="${TEST_SFT_DIR:-$LOCAL_ROOT/train_data/paper/test500}"

mkdir -p "$TRAIN_DATA_ROOT" "$ADAPTER_ROOT" "$RESULTS_ROOT" "$LOG_ROOT" "$HF_CACHE" "$LOCAL_MODEL_ROOT"

source "$LINUX_DIR/model_registry.sh"

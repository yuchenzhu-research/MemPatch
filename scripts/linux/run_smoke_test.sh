#!/usr/bin/env bash
# Lightweight CUDA smoke: 10-step train, resume, pick-best, with/without eval.
#
#   export LOCAL_ROOT=/root/autodl-tmp/mempatch_local
#   export HF_HOME=$LOCAL_ROOT/hf_cache
#   SLUG=llama3_1_8b bash scripts/linux/run_smoke_test.sh
#
# Overrides (optional):
#   TRAIN_ITERS=10 SAVE_EVERY=2 RUN_ID=smoke10 KFOLDS=5 EVAL_LIMIT=20
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$LINUX_DIR/lib_selection.sh"

# Smoke profile always wins over env.sh paper defaults (TRAIN_ITERS=256, etc.).
# Override with SMOKE_* env vars if needed, e.g. SMOKE_TRAIN_ITERS=16.
SLUG="${SLUG:-llama3_1_8b}"
export RUN_ID="${SMOKE_RUN_ID:-smoke10}"
export TRAIN_ITERS="${SMOKE_TRAIN_ITERS:-10}"
export SAVE_EVERY="${SMOKE_SAVE_EVERY:-2}"
export SAVE_TOTAL_LIMIT="${SMOKE_SAVE_TOTAL_LIMIT:-10}"
export KFOLDS="${SMOKE_KFOLDS:-5}"
export EVAL_LIMIT="${SMOKE_EVAL_LIMIT:-20}"
RESUME_PROBE_STEPS="${SMOKE_RESUME_PROBE_STEPS:-4}"
FULL_TRAIN_ITERS="$TRAIN_ITERS"

RESULT_DIR="$RESULTS_ROOT/$SLUG"
SMOKE_SFT_DIR="${LOCAL_ROOT}/train_data/paper/test${EVAL_LIMIT}"

echo "======== CUDA probe ========"
"$PYTHON" - <<'PY'
import sys
import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("error: no CUDA GPU — smoke requires a GPU node (AutoDL PyTorch+CUDA image).", file=sys.stderr)
    sys.exit(1)
print("device:", torch.cuda.get_device_name(0))
PY

echo "======== Hugging Face auth probe ========"
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN="$HF_TOKEN" "$PYTHON" -c 'import os; from huggingface_hub import login; login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)'
  echo "HF_TOKEN applied."
elif ! "$PYTHON" -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
  echo "error: Hugging Face not authenticated." >&2
  echo "  export HF_TOKEN=hf_...  (accept Llama/Gemma licenses on huggingface.co first)" >&2
  echo "  or: hf auth login" >&2
  exit 1
else
  echo "HF session OK."
fi

echo "======== audit ========"
bash "$LINUX_DIR/01_audit.sh"

echo "======== prepare + train (resume probe on fold 0) ========"
KFOLD_FOLD=0 bash "$LINUX_DIR/02_prepare_kfold.sh"

OUT_DIR="$ADAPTER_ROOT/${SLUG}_pathB_lora/fold0/${RUN_ID}"
PARTIAL_ITERS="$RESUME_PROBE_STEPS"
if [[ "$PARTIAL_ITERS" -ge "$TRAIN_ITERS" ]]; then
  PARTIAL_ITERS=$((TRAIN_ITERS / 2))
fi

echo "-- partial train: ${PARTIAL_ITERS} steps -> resume -> ${FULL_TRAIN_ITERS} steps"
TRAIN_ITERS="$PARTIAL_ITERS" KFOLD_FOLD=0 bash "$LINUX_DIR/03_train_fold.sh"
RESUME_FROM="$OUT_DIR/checkpoint-${PARTIAL_ITERS}"
if [[ ! -d "$RESUME_FROM" ]]; then
  echo "error: expected resume checkpoint missing: $RESUME_FROM" >&2
  ls -la "$OUT_DIR" >&2 || true
  exit 1
fi
TRAIN_ITERS="$FULL_TRAIN_ITERS" RESUME_FROM="$RESUME_FROM" KFOLD_FOLD=0 bash "$LINUX_DIR/03_train_fold.sh"

echo "======== remaining folds (full ${FULL_TRAIN_ITERS} steps each) ========"
for fold in $(seq 1 $((KFOLDS - 1))); do
  echo "---- fold $fold ----"
  unset RESUME_FROM
  KFOLD_FOLD="$fold" bash "$LINUX_DIR/02_prepare_kfold.sh"
  KFOLD_FOLD="$fold" bash "$LINUX_DIR/03_train_fold.sh"
done

echo "======== pick best fold + checkpoint ========"
SLUG="$SLUG" bash "$LINUX_DIR/05_pick_best.sh"
BEST_FOLD="$(load_best_fold "$SLUG")"
BEST_CKPT="$(load_best_checkpoint "$SLUG")"
echo "best_fold=$BEST_FOLD"
echo "best_checkpoint=$BEST_CKPT"

echo "======== eval bundle (limit=${EVAL_LIMIT}) ========"
mkdir -p "$SMOKE_SFT_DIR"
"$PYTHON" "$ROOT/scripts/data/build_paper_eval_bundle.py" \
  --scenarios "$TEST_SCENARIOS" \
  --out-dir "$SMOKE_SFT_DIR" \
  --limit "$EVAL_LIMIT"

HF_MODEL="$(resolve_hf_model "$SLUG")"
TEST_SFT_DIR="$SMOKE_SFT_DIR" EVAL_LIMIT="$EVAL_LIMIT" \
  SLUG="$SLUG" bash "$LINUX_DIR/06_eval_test.sh"

echo "======== per-fold with/without (LoRA ckpt @ step ${SAVE_EVERY}) ========"
for fold in $(seq 0 $((KFOLDS - 1))); do
  CKPT_DIR="$ADAPTER_ROOT/${SLUG}_pathB_lora/fold${fold}/${RUN_ID}/checkpoint-${FULL_TRAIN_ITERS}"
  if [[ ! -d "$CKPT_DIR" ]]; then
    CKPT_DIR="$(ls -d "$ADAPTER_ROOT/${SLUG}_pathB_lora/fold${fold}/${RUN_ID}"/checkpoint-* 2>/dev/null | tail -1)"
  fi
  for variant in base lora; do
    tag="smoke_fold${fold}_${variant}"
    extra=(--limit "$EVAL_LIMIT")
    if [[ "$variant" == "base" ]]; then
      extra+=(--no-adapter)
    else
      extra+=(--adapter-path "$CKPT_DIR")
    fi
    "$PYTHON" "$LINUX_DIR/run_hf_test_eval.py" \
      --data "$SMOKE_SFT_DIR/sft.jsonl" \
      --eval-data "$SMOKE_SFT_DIR/scenarios.jsonl" \
      --model-id "$HF_MODEL" \
      --out-predictions "$RESULT_DIR/${tag}_predictions.jsonl" \
      --out-metrics "$RESULT_DIR/${tag}_metrics.json" \
      --split-tag "$tag" \
      --model-tag "$SLUG" \
      --variant-tag "$variant" \
      "${extra[@]}"
  done
done

echo "======== 8+1 baseline matrix (11 baselines + lora_best) ========"
# Build matching SFT bundle for lora_best row in baseline matrix.
"$PYTHON" "$ROOT/scripts/data/build_paper_eval_bundle.py" \
  --scenarios "$TEST_SCENARIOS" \
  --out-dir "$SMOKE_SFT_DIR" \
  --limit "$EVAL_LIMIT"

BASELINE_SET="${BASELINE_SET:-all}" INCLUDE_LORA=1 RESUME=0 \
  SLUG="$SLUG" EVAL_LIMIT="$EVAL_LIMIT" TEST_SFT_DIR="$SMOKE_SFT_DIR" \
  EVAL_SCENARIOS="$SMOKE_SFT_DIR/scenarios.jsonl" \
  bash "$LINUX_DIR/run_baseline_matrix.sh"

echo "Smoke OK — artifacts under $RESULT_DIR and $OUT_DIR"

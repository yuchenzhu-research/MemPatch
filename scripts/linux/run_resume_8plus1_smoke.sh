#!/usr/bin/env bash
# Offline server smoke test for all three paper backbones:
#   1. validate local datasets, local model weights, CUDA, and library APIs;
#   2. train step 1 and save checkpoint-1;
#   3. resume checkpoint-1 and train through step 2;
#   4. run 8 baseline proxies + Path A LoRA/DPA on one test case each.
#
# This is a plumbing/resume test, not a quality experiment.
#
#   export LOCAL_ROOT=/root/autodl-tmp/mempatch_local
#   bash scripts/linux/run_resume_8plus1_smoke.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

SMOKE_RUN_ID="${SMOKE_RUN_ID:-$(date -u '+%Y%m%dT%H%M%SZ')}"
SMOKE_ROOT="${SMOKE_ROOT:-$LOCAL_ROOT/smoke/resume_8plus1/$SMOKE_RUN_ID}"
SMOKE_TRAIN_ROWS="${SMOKE_TRAIN_ROWS:-8}"
SMOKE_VALID_ROWS="${SMOKE_VALID_ROWS:-2}"
SMOKE_FAIL_ON_WARNINGS="${SMOKE_FAIL_ON_WARNINGS:-0}"
SLUGS=(mistral_nemo_12b gemma3_12b qwen3_14b)

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

mkdir -p "$SMOKE_ROOT"
exec > >(tee -a "$SMOKE_ROOT/smoke.log") 2>&1

log() { echo "[$(date '+%F %T')] smoke: $*"; }

TRAIN_SCENARIOS="${TRAIN_SCENARIOS:-$LOCAL_ROOT/data/mempatch/train/scenarios.jsonl}"
TEST_SCENARIOS="${TEST_SCENARIOS:-$LOCAL_ROOT/data/mempatch/test/scenarios.jsonl}"
TEST_SFT_DIR="$SMOKE_ROOT/test_bundle"
export TRAIN_SCENARIOS TEST_SCENARIOS TEST_SFT_DIR

MODEL_ARGS=()
for slug in "${SLUGS[@]}"; do
  hub_id="$(resolve_hf_model_hub "$slug")"
  model_dir="$(local_model_dir_for_hub "$hub_id")"
  MODEL_ARGS+=(--model "$slug=$model_dir")
done

log "preflight (offline; no model or dataset download allowed)"
"$PYTHON" "$LINUX_DIR/smoke_support.py" preflight \
  --out "$SMOKE_ROOT/environment.json" \
  --train-data "$TRAIN_SCENARIOS" \
  --test-data "$TEST_SCENARIOS" \
  "${MODEL_ARGS[@]}"
"$PYTHON" -m pip freeze >"$SMOKE_ROOT/pip-freeze.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q >"$SMOKE_ROOT/nvidia-smi.txt"
fi

log "dataset decision-boundary audit"
AUDIT_TRAIN="$(dirname "$TRAIN_SCENARIOS")" \
  AUDIT_TEST="$(dirname "$TEST_SCENARIOS")" \
  bash "$LINUX_DIR/01_audit.sh"

log "Path A deterministic view -> DPA -> benchmark projection check"
"$PYTHON" "$LINUX_DIR/smoke_support.py" verify-path-a \
  --test-data "$TEST_SCENARIOS" \
  --out "$SMOKE_ROOT/path_a_dpa_report.json"

log "prepare tiny SFT bundle from the production fixed split"
"$PYTHON" "$LINUX_DIR/smoke_support.py" prepare-sft \
  --train-data "$TRAIN_SCENARIOS" \
  --out-dir "$SMOKE_ROOT/sft" \
  --train-rows "$SMOKE_TRAIN_ROWS" \
  --valid-rows "$SMOKE_VALID_ROWS" \
  --split-index "$SPLIT_INDEX" \
  --split-parts "$SPLIT_PARTS" \
  --seed "$SEED"

for slug in "${SLUGS[@]}"; do
  hub_id="$(resolve_hf_model_hub "$slug")"
  model_dir="$(local_model_dir_for_hub "$hub_id")"
  adapter_dir="$SMOKE_ROOT/adapters/$slug"
  train_log_dir="$SMOKE_ROOT/train_logs/$slug"
  result_root="$SMOKE_ROOT/eval"
  model_log="$SMOKE_ROOT/${slug}.log"

  mkdir -p "$adapter_dir" "$train_log_dir" "$result_root"
  log "[$slug] step 1: fresh train and checkpoint save"
  "$PYTHON" "$LINUX_DIR/train_qlora.py" \
    --model-id "$model_dir" \
    --train-data "$SMOKE_ROOT/sft/train.jsonl" \
    --valid-data "$SMOKE_ROOT/sft/valid.jsonl" \
    --output-dir "$adapter_dir" \
    --log-dir "$train_log_dir" \
    --max-steps 1 \
    --save-steps 1 \
    --eval-steps 1 \
    --save-total-limit 2 \
    --max-seq-length 2048 \
    --seed "$SEED" 2>&1 | tee -a "$model_log"

  [[ -f "$adapter_dir/checkpoint-1/trainer_state.json" ]] || {
    echo "error: $slug did not save checkpoint-1" >&2
    exit 1
  }

  log "[$slug] step 2: resume checkpoint-1"
  "$PYTHON" "$LINUX_DIR/train_qlora.py" \
    --model-id "$model_dir" \
    --train-data "$SMOKE_ROOT/sft/train.jsonl" \
    --valid-data "$SMOKE_ROOT/sft/valid.jsonl" \
    --output-dir "$adapter_dir" \
    --log-dir "$train_log_dir" \
    --max-steps 2 \
    --save-steps 1 \
    --eval-steps 1 \
    --save-total-limit 2 \
    --max-seq-length 2048 \
    --seed "$SEED" \
    --resume-from-checkpoint "$adapter_dir/checkpoint-1" 2>&1 | tee -a "$model_log"

  "$PYTHON" "$LINUX_DIR/smoke_support.py" verify-resume \
    --output-dir "$adapter_dir" \
    --log-dir "$train_log_dir" \
    --out "$SMOKE_ROOT/${slug}_resume_report.json"

  log "[$slug] 8 baseline proxies + Path A LoRA/DPA, one case each"
  BEST_CHECKPOINT="$adapter_dir/checkpoint-2" \
    OUTPUT_ROOT="$result_root" \
    EVAL_LIMIT=1 \
    EVAL_PREFIX=smoke1 \
    SKIP_BASE=1 \
    PATH_A_STRICT_SMOKE=0 \
    SLUG="$slug" \
    bash "$LINUX_DIR/run_eval_subset.sh" 2>&1 | tee -a "$model_log"

  "$PYTHON" "$LINUX_DIR/smoke_support.py" verify-eval \
    --result-dir "$result_root/$slug" \
    --prefix smoke1 \
    --out "$SMOKE_ROOT/${slug}_eval_report.json"
done

grep -Eini 'DeprecationWarning|FutureWarning|deprecated|unexpected keyword|will be removed' \
  "$SMOKE_ROOT"/*.log >"$SMOKE_ROOT/warnings.txt" || true

if [[ -s "$SMOKE_ROOT/warnings.txt" ]]; then
  log "warnings found; inspect $SMOKE_ROOT/warnings.txt"
  cat "$SMOKE_ROOT/warnings.txt"
  if [[ "$SMOKE_FAIL_ON_WARNINGS" == "1" ]]; then
    exit 1
  fi
else
  log "no deprecation/future-warning patterns found"
fi

date -Iseconds >"$SMOKE_ROOT/smoke.done"
log "PASS -> $SMOKE_ROOT"

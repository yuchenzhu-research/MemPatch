#!/usr/bin/env bash
# One-model paper pipeline with resumable phases.
#
#   SLUG=mistral_nemo_12b bash scripts/linux/run_model.sh
#   SLUG=gemma3_12b PHASES=prefetch,train,pick,eval,baselines bash scripts/linux/run_model.sh
#   SLUG=mistral_nemo_12b PHASES=eval,baselines bash scripts/linux/run_model.sh
#
# PHASES=auto (default): run any phase not yet complete.
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$LINUX_DIR/lib_phases.sh"

SLUG="${SLUG:?set SLUG}"
PHASES="${PHASES:-auto}"
PIPELINE_LOG="${PIPELINE_LOG:-$LOCAL_ROOT/logs/pipeline.log}"
mkdir -p "$(dirname "$PIPELINE_LOG")" "$LOCAL_ROOT/models"

log() {
  echo "[$(date '+%F %T')] [$SLUG] $*" | tee -a "$PIPELINE_LOG"
}

want_phase() {
  local name="$1"
  if [[ "$PHASES" == "auto" ]]; then
    case "$name" in
      prefetch) phase_prefetch_done "$SLUG" || return 0 ;;
      train) phase_train_done "$SLUG" || return 0 ;;
      pick) phase_pick_done "$SLUG" || return 0 ;;
      eval) phase_eval_done "$SLUG" || return 0 ;;
      baselines) phase_baselines_done "$SLUG" || return 0 ;;
    esac
    return 1
  fi
  [[ ",$PHASES," == *",$name,"* ]]
}

run_prefetch() {
  log "prefetch start"
  bash "$LINUX_DIR/prefetch_model.sh"
  log "prefetch done -> $(resolve_hf_model "$SLUG")"
}

run_train() {
  log "train 5-fold start"
  for fold in $(seq 0 $((KFOLDS - 1))); do
    if [[ -f "$LOG_ROOT/${SLUG}_fold${fold}/trainer_metrics.json" ]]; then
      log "fold $fold already trained, skip"
      continue
    fi
    log "fold $fold prepare+train"
    KFOLD_FOLD="$fold" bash "$LINUX_DIR/02_prepare_kfold.sh"
    KFOLD_FOLD="$fold" bash "$LINUX_DIR/03_train_fold.sh"
  done
  log "train 5-fold done"
}

run_pick() {
  log "pick best fold + checkpoint (${KFOLDS} folds x $((TRAIN_ITERS / SAVE_EVERY)) ckpts)"
  SLUG="$SLUG" bash "$LINUX_DIR/05_pick_best.sh" | tee -a "$PIPELINE_LOG"
  log "pick done"
}

run_eval() {
  log "eval test500 without + with LoRA"
  SLUG="$SLUG" bash "$LINUX_DIR/06_eval_test.sh" | tee -a "$PIPELINE_LOG"
  log "eval done"
}

run_baselines() {
  log "baselines 11 + mempatch_lora_best"
  SLUG="$SLUG" RESUME=1 bash "$LINUX_DIR/run_baseline_matrix.sh" | tee -a "$PIPELINE_LOG"
  log "baselines done"
}

log "===== pipeline start phases=$PHASES ====="
print_model_status "$SLUG" | tee -a "$PIPELINE_LOG"

if want_phase prefetch; then run_prefetch; fi
if want_phase train; then run_train; fi
if want_phase pick; then run_pick; fi
if want_phase eval; then run_eval; fi
if want_phase baselines; then run_baselines; fi

log "===== pipeline end ====="
print_model_status "$SLUG" | tee -a "$PIPELINE_LOG"

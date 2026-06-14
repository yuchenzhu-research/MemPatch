#!/usr/bin/env bash
# Smoke/probe pipeline for phi4_14b. Trains 10 steps and runs subset eval.
#
#   bash scripts/linux/run_phi4_smoke10.sh
#   PHASES=eval,baselines bash scripts/linux/run_phi4_smoke10.sh
#
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$LINUX_DIR/lib_phases.sh"
source "$LINUX_DIR/lib_gpu.sh"

export SLUG="phi4_14b"
export RUN_ID="phi4_smoke10"
export TRAIN_ITERS=10
export SAVE_EVERY=5
export SAVE_TOTAL_LIMIT=2
export SPLIT_INDEX="${SPLIT_INDEX:-0}"
export SEED="${SEED:-42}"
export EVAL_LIMIT="${EVAL_LIMIT:-25}"

PHASES="${PHASES:-auto}"
PIPELINE_LOG="${PIPELINE_LOG:-$LOCAL_ROOT/logs/pipeline.log}"
mkdir -p "$(dirname "$PIPELINE_LOG")" "$LOCAL_ROOT/models"

log() {
  echo "[$(date '+%F %T')] [$SLUG] smoke10: $*" | tee -a "$PIPELINE_LOG"
}

phase_phi4_eval_done() {
  [[ -f "$RESULTS_ROOT/$SLUG/phi4_smoke10_base_predictions.jsonl" ]] \
    && [[ -f "$RESULTS_ROOT/$SLUG/phi4_smoke10_lora_best_predictions.jsonl" ]] \
    && [[ -f "$RESULTS_ROOT/$SLUG/phi4_smoke10_lora_best_manifest.json" ]] \
    && [[ -f "$RESULTS_ROOT/$SLUG/phi4_smoke10_path_a_lora_best_predictions.jsonl" ]] \
    && [[ -f "$RESULTS_ROOT/$SLUG/phi4_smoke10_path_a_lora_best_manifest.json" ]] \
    && [[ -f "$RESULTS_ROOT/$SLUG/phi4_smoke10_path_a_lora_best_no_dpa_manifest.json" ]]
}

phase_phi4_baselines_done() {
  local b
  for b in no_memory memory_only prompt_only model_only mempatch_only mempatch_no_gate mempatch_no_dpa; do
    [[ -f "$RESULTS_ROOT/$SLUG/phi4_smoke10_baseline_${b}_predictions.jsonl" ]] || return 1
  done
  return 0
}

want_phase() {
  local name="$1"
  if [[ "$PHASES" == "auto" ]]; then
    case "$name" in
      prefetch) phase_prefetch_done "$SLUG" || return 0 ;;
      train) phase_train_done "$SLUG" || return 0 ;;
      pick) phase_pick_done "$SLUG" || return 0 ;;
      eval) phase_phi4_eval_done || return 0 ;;
      baselines) phase_phi4_baselines_done || return 0 ;;
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
  local metrics="$LOG_ROOT/${SLUG}_split${SPLIT_INDEX}/${RUN_ID}/trainer_metrics.json"
  log "train start (steps=$TRAIN_ITERS)"
  if [[ -f "$metrics" ]]; then
    log "train metrics present, skipping SFT step"
    return
  fi
  bash "$LINUX_DIR/02_prepare_split.sh"
  bash "$LINUX_DIR/03_train.sh"
  release_gpu
  log "train done"
}

run_pick() {
  log "pick checkpoint (steps=$TRAIN_ITERS)"
  bash "$LINUX_DIR/05_pick_best.sh" | tee -a "$PIPELINE_LOG"
  log "pick done"
}

run_eval() {
  release_gpu
  log "eval start (EVAL_LIMIT=$EVAL_LIMIT)"
  EVAL_PREFIX=phi4_smoke10 bash "$LINUX_DIR/06_eval_test.sh" | tee -a "$PIPELINE_LOG"
  EVAL_PREFIX=phi4_smoke10_path_a bash "$LINUX_DIR/07_eval_path_a.sh" --variant lora | tee -a "$PIPELINE_LOG"
  release_gpu
  log "eval done"
}

run_baselines() {
  release_gpu
  log "baselines start (EVAL_LIMIT=$EVAL_LIMIT)"
  BASELINE_SET=main \
  INCLUDE_LORA=0 \
  PRED_TAG_PREFIX=phi4_smoke10_baseline_ \
  RESUME=0 \
  bash "$LINUX_DIR/run_baseline_matrix.sh" | tee -a "$PIPELINE_LOG"
  log "baselines done"
}

log "===== Phi-4 smoke10 pipeline start ====="

if want_phase prefetch; then run_prefetch; fi
if want_phase train; then run_train; fi
if want_phase pick; then run_pick; fi
if want_phase eval; then run_eval; fi
if want_phase baselines; then run_baselines; fi

release_gpu
log "===== Phi-4 smoke10 pipeline end ====="
exit 0

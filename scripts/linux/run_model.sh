#!/usr/bin/env bash
# One-model paper pipeline with resumable phases.
#
#   SLUG=mistral_nemo_12b bash scripts/linux/run_model.sh
#   SLUG=gemma3_12b PHASES=prefetch,train,pick,eval,baselines bash scripts/linux/run_model.sh
#   SLUG=mistral_nemo_12b PHASES=smoke bash scripts/linux/run_model.sh  # optional manual only
#
# PHASES=auto (default): run any phase not yet complete.
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$LINUX_DIR/lib_phases.sh"
source "$LINUX_DIR/lib_gpu.sh"

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
      smoke) return 1 ;;
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
  local split_idx="$SPLIT_INDEX"
  local metrics="$LOG_ROOT/${SLUG}_split${split_idx}/${RUN_ID}/trainer_metrics.json"
  log "train start (held-out partition $split_idx/$SPLIT_PARTS, steps=$TRAIN_ITERS)"
  if [[ -f "$metrics" ]]; then
    log "train artifacts present, skip"
    return
  fi
  SPLIT_INDEX="$split_idx" bash "$LINUX_DIR/02_prepare_split.sh"
  SPLIT_INDEX="$split_idx" bash "$LINUX_DIR/03_train.sh"
  release_gpu
  log "train done"
}

run_pick() {
  log "pick checkpoint on fixed L3 val partition ($((TRAIN_ITERS / SAVE_EVERY)) candidates)"
  SLUG="$SLUG" bash "$LINUX_DIR/05_pick_best.sh" | tee -a "$PIPELINE_LOG"
  log "pick done"
}

run_smoke() {
  log "smoke: Path B w/o + w/ LoRA and public-data baselines (${SMOKE_LIMIT} case each)"
  SLUG="$SLUG" bash "$LINUX_DIR/run_smoke.sh" | tee -a "$PIPELINE_LOG"
  log "smoke done"
}

run_eval() {
  release_gpu
  log "Path B direct-response test500 without + with LoRA (no DPA ablation)"
  unset EVAL_LIMIT
  EVAL_PREFIX=test500 SLUG="$SLUG" bash "$LINUX_DIR/06_eval_test.sh" | tee -a "$PIPELINE_LOG"
  log "Path A LoRA test500 (paired typed-actions: full DPA + no-DPA ablation)"
  EVAL_PREFIX=test500_path_a SLUG="$SLUG" bash "$LINUX_DIR/07_eval_path_a.sh" --variant lora | tee -a "$PIPELINE_LOG"
  release_gpu
  log "Path A + Path B eval done"
}

run_baselines() {
  release_gpu
  log "public-data paper baselines on test500 (BASELINE_SET=${BASELINE_SET:-main})"
  unset EVAL_LIMIT
  BASELINE_SET="${BASELINE_SET:-main}" INCLUDE_LORA=0 \
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
if [[ "$PHASES" != "auto" ]] && want_phase smoke; then run_smoke; fi

release_gpu
log "===== pipeline end ====="
print_model_status "$SLUG" | tee -a "$PIPELINE_LOG"

#!/usr/bin/env bash
# One-shot 1024-step campaign in fixed backbone order. Each model runs:
# prefetch → fixed-split train → checkpoint pick → Path A DPA/no-DPA + Path B → baselines.
# (smoke is manual-only: PHASES=smoke bash scripts/linux/run_model.sh)
#
#   export LOCAL_ROOT=/root/autodl-tmp/mempatch_local
#   bash scripts/linux/run_paper_campaign.sh
#
# Background: bash scripts/linux/start_background.sh
set -euo pipefail

# Set campaign defaults before env.sh installs its general-purpose defaults.
# Callers may still override these values explicitly.
export RUN_ID="${RUN_ID:-full1024}"
export TRAIN_ITERS="${TRAIN_ITERS:-1024}"
export SAVE_EVERY="${SAVE_EVERY:-128}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"

source "$(dirname "$0")/env.sh"
source "$LINUX_DIR/lib_phases.sh"
source "$LINUX_DIR/lib_gpu.sh"

PIPELINE_LOG="${PIPELINE_LOG:-$LOCAL_ROOT/logs/pipeline.log}"
mkdir -p "$(dirname "$PIPELINE_LOG")"

# Complete higher-priority models first so an interrupted allocation still
# leaves their selected checkpoints and formal test results intact.
SLUGS=("${FORMAL_SLUGS[@]}")

log() {
  echo "[$(date '+%F %T')] campaign: $*" | tee -a "$PIPELINE_LOG"
}

if (( TRAIN_ITERS % SAVE_EVERY != 0 )); then
  log "ERROR: TRAIN_ITERS=$TRAIN_ITERS must be divisible by SAVE_EVERY=$SAVE_EVERY"
  exit 2
fi
required_checkpoints=$((TRAIN_ITERS / SAVE_EVERY))
if (( SAVE_TOTAL_LIMIT < required_checkpoints )); then
  log "ERROR: SAVE_TOTAL_LIMIT=$SAVE_TOTAL_LIMIT would discard some of the $required_checkpoints checkpoint candidates"
  exit 2
fi

log "start order: ${SLUGS[*]} (steps=$TRAIN_ITERS, checkpoint_every=$SAVE_EVERY, PHASES=auto per model)"

if ! AUDIT_TRAIN="$(resolve_split_dir train)" || ! AUDIT_TEST="$(resolve_split_dir test)"; then
  log "ERROR: full train/test scenarios not found."
  log "  Place dataset at $LOCAL_ROOT/data/mempatch/{train,test}/scenarios.jsonl"
  exit 1
fi
log "audit train=$AUDIT_TRAIN test=$AUDIT_TEST"
AUDIT_TRAIN="$AUDIT_TRAIN" AUDIT_TEST="$AUDIT_TEST" bash "$LINUX_DIR/01_audit.sh" 2>&1 | tee -a "$PIPELINE_LOG"

for slug in "${SLUGS[@]}"; do
  log "===== $slug (auto: skip finished phases) ====="
  print_model_status "$slug" | tee -a "$PIPELINE_LOG"
  SLUG="$slug" PHASES=auto bash "$LINUX_DIR/run_model.sh" 2>&1 | tee -a "$PIPELINE_LOG"
  print_model_status "$slug" | tee -a "$PIPELINE_LOG"
  release_gpu
done

log "campaign complete. Results: $RESULTS_ROOT/"

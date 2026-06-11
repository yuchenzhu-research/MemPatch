#!/usr/bin/env bash
# One-shot paper pipeline in fixed backbone order. Each model runs:
# prefetch → fixed-split train → checkpoint pick → Path A DPA/no-DPA + Path B → baselines.
# (smoke is manual-only: PHASES=smoke bash scripts/linux/run_model.sh)
#
#   export LOCAL_ROOT=/root/autodl-tmp/mempatch_local
#   bash scripts/linux/run_paper_campaign.sh
#
# Background: bash scripts/linux/start_background.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$LINUX_DIR/lib_phases.sh"
source "$LINUX_DIR/lib_gpu.sh"

PIPELINE_LOG="${PIPELINE_LOG:-$LOCAL_ROOT/logs/pipeline.log}"
mkdir -p "$(dirname "$PIPELINE_LOG")"

# Fixed backbone order — do not reorder without editing this script.
SLUGS=(mistral_nemo_12b gemma3_12b qwen3_14b)

log() {
  echo "[$(date '+%F %T')] campaign: $*" | tee -a "$PIPELINE_LOG"
}

log "start order: ${SLUGS[*]} (PHASES=auto per model)"

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

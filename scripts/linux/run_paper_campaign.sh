#!/usr/bin/env bash
# One-shot paper pipeline (fixed order):
#   1. mistral_nemo_12b  — 8+1 on test500 (skips done train/pick/smoke)
#   2. gemma3_12b        — train 5-fold → smoke → 8+1
#   3. qwen3_14b         — train 5-fold → smoke → 8+1
#
#   export LOCAL_ROOT=/root/autodl-tmp/mempatch_local
#   export HF_TOKEN=hf_...
#   bash scripts/linux/run_paper_campaign.sh
#
# Background: bash scripts/linux/start_background.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$LINUX_DIR/lib_phases.sh"

PIPELINE_LOG="${PIPELINE_LOG:-$LOCAL_ROOT/logs/pipeline.log}"
mkdir -p "$(dirname "$PIPELINE_LOG")"

# Fixed backbone order — do not reorder without editing this script.
SLUGS=(mistral_nemo_12b gemma3_12b qwen3_14b)

log() {
  echo "[$(date '+%F %T')] campaign: $*" | tee -a "$PIPELINE_LOG"
}

log "start order: ${SLUGS[*]} (PHASES=auto per model)"

if AUDIT_TRAIN="$(resolve_split_dir train)" && AUDIT_TEST="$(resolve_split_dir test)"; then
  log "audit train=$AUDIT_TRAIN test=$AUDIT_TEST"
  AUDIT_TRAIN="$AUDIT_TRAIN" AUDIT_TEST="$AUDIT_TEST" bash "$LINUX_DIR/01_audit.sh" 2>&1 | tee -a "$PIPELINE_LOG"
else
  log "WARN: audit skipped — full train/test scenarios not found."
  log "  Run once: cp -a ../MemPatch.bak/hf_release/mempatch $LOCAL_ROOT/data/"
  log "  Or: python scripts/data/generate_mempatch.py --full --out-dir $LOCAL_ROOT/data/mempatch"
fi

for slug in "${SLUGS[@]}"; do
  log "===== $slug (auto: skip finished phases) ====="
  print_model_status "$slug" | tee -a "$PIPELINE_LOG"
  SLUG="$slug" PHASES=auto bash "$LINUX_DIR/run_model.sh" 2>&1 | tee -a "$PIPELINE_LOG"
  print_model_status "$slug" | tee -a "$PIPELINE_LOG"
done

log "campaign complete. Results: $RESULTS_ROOT/"

#!/usr/bin/env bash
# Paper campaign: mistral experiments first, then train+eval gemma+qwen.
#
# Mistral (train already done): smoke -> full MemPatch w/o+w/ LoRA -> 8 baselines
# Gemma + Qwen: full 5-fold train -> smoke -> full eval -> 8 baselines
#
#   export LOCAL_ROOT=/root/autodl-tmp/mempatch_local
#   export HF_TOKEN=hf_...
#   bash scripts/linux/run_paper_campaign.sh
#
# Env:
#   EXPERIMENT_SLUGS   default: mistral_nemo_12b
#   TRAIN_SLUGS        default: gemma3_12b qwen3_14b
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$LINUX_DIR/lib_phases.sh"

PIPELINE_LOG="${PIPELINE_LOG:-$LOCAL_ROOT/logs/pipeline.log}"
mkdir -p "$(dirname "$PIPELINE_LOG")"

EXPERIMENT_SLUGS=(${EXPERIMENT_SLUGS:-mistral_nemo_12b})
TRAIN_SLUGS=(${TRAIN_SLUGS:-gemma3_12b qwen3_14b})

log() {
  echo "[$(date '+%F %T')] campaign: $*" | tee -a "$PIPELINE_LOG"
}

run_experiment_only() {
  local slug="$1"
  log "===== $slug: skip train if done; smoke -> eval -> baselines ====="
  SLUG="$slug" PHASES=smoke,eval,baselines bash "$LINUX_DIR/run_model.sh"
}

run_full_pipeline() {
  local slug="$1"
  log "===== $slug: prefetch/train/pick -> smoke -> eval -> baselines ====="
  SLUG="$slug" PHASES=auto bash "$LINUX_DIR/run_model.sh"
}

log "start EXPERIMENT_SLUGS=${EXPERIMENT_SLUGS[*]} TRAIN_SLUGS=${TRAIN_SLUGS[*]}"

bash "$LINUX_DIR/01_audit.sh" 2>&1 | tee -a "$PIPELINE_LOG"

for slug in "${EXPERIMENT_SLUGS[@]}"; do
  run_experiment_only "$slug"
done

for slug in "${TRAIN_SLUGS[@]}"; do
  run_full_pipeline "$slug"
done

log "campaign complete. Results: $RESULTS_ROOT/"

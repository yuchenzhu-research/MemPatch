#!/usr/bin/env bash
# Destructive, explicit paper rerun for the two anomalous backbones.
set -euo pipefail
source "$(dirname "$0")/env.sh"

if [[ "${CONFIRM_RERUN:-0}" != "1" ]]; then
  echo "Set CONFIRM_RERUN=1 to delete Qwen/Mistral artifacts for RUN_ID=$RUN_ID" >&2
  exit 2
fi

slugs=(qwen3_14b mistral_nemo_12b)
for slug in "${slugs[@]}"; do
  adapter_dir="$ADAPTER_ROOT/${slug}_multitask_lora/split${SPLIT_INDEX}/${RUN_ID}"
  log_dir="$LOG_ROOT/${slug}_split${SPLIT_INDEX}/${RUN_ID}"
  result_dir="$RESULTS_ROOT/$slug"
  printf 'Removing exact-run artifacts:\n  %s\n  %s\n  %s\n' \
    "$adapter_dir" "$log_dir" "$result_dir"
  rm -rf -- "$adapter_dir" "$log_dir" "$result_dir"
done

for slug in "${slugs[@]}"; do
  SLUG="$slug" PHASES=train,pick,eval,baselines \
    bash "$LINUX_DIR/run_model.sh"
done

"$PYTHON" "$LINUX_DIR/diagnose_result_bundle.py" \
  --results-root "$RESULTS_ROOT" \
  --slugs "${slugs[@]}"

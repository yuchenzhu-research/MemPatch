#!/usr/bin/env bash
# Fresh two-stage paper campaign:
#   1. train + pick all three backbones;
#   2. run the configured frozen baseline set + MemPatch on all 500 test cases.
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$LINUX_DIR/lib_gpu.sh"

SLUGS=(mistral_nemo_12b gemma3_12b qwen3_14b)
CAMPAIGN_LOG="${CAMPAIGN_LOG:-$LOCAL_ROOT/logs/train_all_then_7plus1.log}"

log() {
  echo "[$(date '+%F %T')] campaign-7plus1: $*" | tee -a "$CAMPAIGN_LOG"
}

if [[ "${CONFIRM_FRESH:-0}" != "1" ]]; then
  echo "Set CONFIRM_FRESH=1 to delete exact-run training artifacts and model result directories." >&2
  echo "RUN_ID=$RUN_ID SPLIT_INDEX=$SPLIT_INDEX" >&2
  exit 2
fi

mkdir -p "$(dirname "$CAMPAIGN_LOG")"
resolve_train_scenarios
resolve_test_scenarios

"$PYTHON" - <<'PY'
import importlib
import sys

required = ("torch", "transformers", "accelerate", "peft", "trl", "bitsandbytes", "datasets")
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    print("error: Linux training dependencies are missing or broken:", file=sys.stderr)
    for row in missing:
        print(f"  {row}", file=sys.stderr)
    print("Run: bash scripts/linux/00_setup.sh", file=sys.stderr)
    raise SystemExit(1)

import torch
if not torch.cuda.is_available():
    raise SystemExit("error: torch is installed but CUDA is unavailable")
print(f"CUDA preflight OK: {torch.cuda.get_device_name(0)}; torch={torch.__version__}")
PY

"$PYTHON" - "$TRAIN_SCENARIOS" "$TEST_SCENARIOS" <<'PY'
import sys
from pathlib import Path

expected = ((Path(sys.argv[1]), 3500, "train"), (Path(sys.argv[2]), 500, "test"))
for path, count, label in expected:
    actual = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    if actual != count:
        raise SystemExit(f"error: {label} must contain {count} scenarios, found {actual}: {path}")
    print(f"{label}: {actual} scenarios -> {path}")
PY

log "audit train=$TRAIN_SCENARIOS test=$TEST_SCENARIOS"
AUDIT_TRAIN="$(dirname "$TRAIN_SCENARIOS")" \
AUDIT_TEST="$(dirname "$TEST_SCENARIOS")" \
  bash "$LINUX_DIR/01_audit.sh" 2>&1 | tee -a "$CAMPAIGN_LOG"

log "clean exact RUN_ID=$RUN_ID artifacts"
for slug in "${SLUGS[@]}"; do
  adapter_dir="$ADAPTER_ROOT/${slug}_multitask_lora/split${SPLIT_INDEX}/${RUN_ID}"
  log_dir="$LOG_ROOT/${slug}_split${SPLIT_INDEX}/${RUN_ID}"
  result_dir="$RESULTS_ROOT/$slug"
  log "remove slug=$slug adapter=$adapter_dir log=$log_dir results=$result_dir"
  rm -rf -- "$adapter_dir" "$log_dir" "$result_dir"
done

log "stage 1/2: train and select checkpoints for all backbones"
for slug in "${SLUGS[@]}"; do
  log "train start slug=$slug steps=$TRAIN_ITERS max_seq_length=$(train_max_seq_length_for_slug "$slug")"
  SLUG="$slug" PHASES=train,pick bash "$LINUX_DIR/run_model.sh" 2>&1 | tee -a "$CAMPAIGN_LOG"
  log "train and pick complete slug=$slug"
  release_gpu
done

log "all three training runs complete; stage 2/2 starts"
unset EVAL_LIMIT
for slug in "${SLUGS[@]}"; do
  log "7+1 test500 start slug=$slug"

  BASELINE_SET=main INCLUDE_LORA=0 RESUME=0 \
    SLUG="$slug" bash "$LINUX_DIR/run_baseline_matrix.sh" 2>&1 | tee -a "$CAMPAIGN_LOG"
  release_gpu

  EVAL_PREFIX=test500_path_a SLUG="$slug" \
    bash "$LINUX_DIR/07_eval_path_a.sh" --variant lora 2>&1 | tee -a "$CAMPAIGN_LOG"
  release_gpu

  "$PYTHON" "$LINUX_DIR/diagnose_result_bundle.py" \
    --results-root "$RESULTS_ROOT" --slugs "$slug" --examples 2 \
    2>&1 | tee -a "$CAMPAIGN_LOG"
  log "7+1 test500 complete slug=$slug"
done

log "campaign complete: results=$RESULTS_ROOT"

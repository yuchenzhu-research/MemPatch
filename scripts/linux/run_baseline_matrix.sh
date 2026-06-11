#!/usr/bin/env bash
# Run paper baseline matrix on Linux HF (default: public-data main baselines).
# Path B with/without LoRA is run via 06_eval_test.sh (not duplicated here).
#
#   SLUG=mistral_nemo_12b EVAL_LIMIT=20 bash scripts/linux/run_baseline_matrix.sh
#
# Env:
#   BASELINE_SET=paper|all|main   (default: main = 7 public-data baselines)
#   INCLUDE_LORA=1               (default: 1 — run lora_best row)
#   RESUME=1                     resume incremental JSONL per baseline
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$LINUX_DIR/lib_selection.sh"

SLUG="${SLUG:?set SLUG}"
HF_MODEL="$(resolve_hf_model "$SLUG")"
RESULT_DIR="${SMOKE_RESULT_DIR:-$RESULTS_ROOT/$SLUG}"
BASELINE_SET="${BASELINE_SET:-main}"
INCLUDE_LORA="${INCLUDE_LORA:-0}"
RESUME="${RESUME:-0}"
PRED_TAG_PREFIX="${PRED_TAG_PREFIX:-baseline_}"
resolve_test_scenarios
EVAL_SCENARIOS="${EVAL_SCENARIOS:-${TEST_SCENARIOS}}"
if [[ -n "${TEST_SFT_DIR:-}" && -f "${TEST_SFT_DIR}/scenarios.jsonl" ]]; then
  EVAL_SCENARIOS="$TEST_SFT_DIR/scenarios.jsonl"
fi

mkdir -p "$RESULT_DIR"

mapfile -t BASELINES < <("$PYTHON" - <<'PY'
import os
from scripts.memory.context_builders import (
    BASELINE_IDS,
    PAPER_APPENDIX_BASELINE_IDS,
    PAPER_MAIN_BASELINE_IDS,
)

mode = os.environ.get("BASELINE_SET", "all")
if mode == "paper" or mode == "main":
    ids = PAPER_MAIN_BASELINE_IDS
elif mode == "all":
    ids = BASELINE_IDS
else:
    ids = tuple(x.strip() for x in mode.split(",") if x.strip())
for bid in ids:
    print(bid)
PY
)

RUN_ONE() {
  local baseline="$1"
  local tag="${PRED_TAG_PREFIX}${baseline}"
  local pred="$RESULT_DIR/${tag}_predictions.jsonl"
  local metrics="$RESULT_DIR/${tag}_metrics.json"
  local extra=()
  if [[ -n "${EVAL_LIMIT:-}" ]]; then
    extra+=(--limit "$EVAL_LIMIT")
  fi
  if [[ "$RESUME" == "1" ]]; then
    extra+=(--resume)
  fi
  echo "===== baseline: $baseline ====="
  "$PYTHON" "$LINUX_DIR/run_hf_baselines.py" \
    --baseline "$baseline" \
    --eval-data "$EVAL_SCENARIOS" \
    --model-id "$HF_MODEL" \
    --out-predictions "$pred" \
    --out-metrics "$metrics" \
    --split-tag "$tag" \
    --model-tag "$SLUG" \
    "${extra[@]}"
}

for baseline in "${BASELINES[@]}"; do
  RUN_ONE "$baseline"
done

if [[ "$INCLUDE_LORA" == "1" ]]; then
  BEST_CKPT="$(ensure_selection "$SLUG")"
  tag="mempatch_lora_best"
  pred="$RESULT_DIR/${tag}_predictions.jsonl"
  metrics="$RESULT_DIR/${tag}_metrics.json"
  extra=(--adapter-path "$BEST_CKPT")
  if [[ -n "${EVAL_LIMIT:-}" ]]; then
    extra+=(--limit "$EVAL_LIMIT")
  fi
  SFT_DATA="${TEST_SFT_DIR:-$LOCAL_ROOT/train_data/paper/test500}/sft.jsonl"
  if [[ ! -f "$SFT_DATA" ]]; then
    echo "Building LoRA eval SFT bundle -> ${TEST_SFT_DIR:-$LOCAL_ROOT/train_data/paper/test500}"
    mkdir -p "${TEST_SFT_DIR:-$LOCAL_ROOT/train_data/paper/test500}"
    "$PYTHON" "$ROOT/scripts/data/build_paper_eval_bundle.py" \
      --scenarios "$EVAL_SCENARIOS" \
      --out-dir "${TEST_SFT_DIR:-$LOCAL_ROOT/train_data/paper/test500}"
  fi
  echo "===== +1 method: mempatch_lora (best checkpoint) ====="
  "$PYTHON" "$LINUX_DIR/run_hf_test_eval.py" \
    --data "$SFT_DATA" \
    --eval-data "$EVAL_SCENARIOS" \
    --model-id "$HF_MODEL" \
    --out-predictions "$pred" \
    --out-metrics "$metrics" \
    --split-tag "$tag" \
    --model-tag "$SLUG" \
    --variant-tag lora_best \
    "${extra[@]}"
fi

if [[ -f "$LINUX_DIR/aggregate_baseline_table.py" ]]; then
  "$PYTHON" "$LINUX_DIR/aggregate_baseline_table.py" \
    --results-dir "$RESULT_DIR" \
    --out "$RESULT_DIR/baseline_matrix.md"
  echo "Wrote $RESULT_DIR/baseline_matrix.md"
fi

if [[ -z "${EVAL_LIMIT:-}" ]]; then
  date -Iseconds >"$RESULT_DIR/baselines_full.done"
fi

echo "Baseline matrix done: ${#BASELINES[@]} baselines (set=${BASELINE_SET}) + lora=${INCLUDE_LORA}"

#!/usr/bin/env bash
# Path A: direct response + typed-action proposer + RevisionGate/DPA projection.
#
#   SLUG=gemma3_12b bash scripts/linux/07_eval_path_a.sh
#   SLUG=gemma3_12b bash scripts/linux/07_eval_path_a.sh --variant base
set -euo pipefail
source "$(dirname "$0")/env.sh"

VARIANT_FILTER="lora"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      VARIANT_FILTER="${2:?base, lora, or all}"
      shift 2
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

SLUG="${SLUG:?set SLUG}"
EVAL_PREFIX="${EVAL_PREFIX:-test500_path_a}"
HF_MODEL="$(resolve_hf_model "$SLUG")"
RESULT_DIR="${SMOKE_RESULT_DIR:-$RESULTS_ROOT/$SLUG}"
mkdir -p "$RESULT_DIR"
resolve_test_scenarios

if [[ ! -f "$TEST_SFT_DIR/sft.jsonl" ]]; then
  "$PYTHON" "$ROOT/scripts/data/build_paper_eval_bundle.py" \
    --scenarios "$TEST_SCENARIOS" \
    --out-dir "$TEST_SFT_DIR"
fi

EVAL_SCENARIOS="$TEST_SCENARIOS"
if [[ -f "$TEST_SFT_DIR/scenarios.jsonl" ]]; then
  EVAL_SCENARIOS="$TEST_SFT_DIR/scenarios.jsonl"
fi

source "$LINUX_DIR/lib_selection.sh"
ADAPTER_PATH=""
if [[ "$VARIANT_FILTER" == "lora" || "$VARIANT_FILTER" == "all" ]]; then
  if [[ -z "${BEST_CHECKPOINT:-}" ]]; then
    BEST_CHECKPOINT="$(ensure_selection "$SLUG")" || {
      echo "hint: SLUG=$SLUG bash scripts/linux/status_checkpoint.sh" >&2
      exit 1
    }
  fi
  ADAPTER_PATH="${BEST_CHECKPOINT:?missing checkpoint}"
fi

run_variant() {
  local variant="$1"
  shift
  local run_tag="${EVAL_PREFIX}_${variant}"
  local pred="$RESULT_DIR/${run_tag}_predictions.jsonl"
  local extra=("$@")
  local args=(
    --data "$TEST_SFT_DIR/sft.jsonl"
    --eval-data "$EVAL_SCENARIOS"
    --model-id "$HF_MODEL"
    --out-predictions "$pred"
    --split-tag "$run_tag"
    --model-tag "$SLUG"
    --variant-tag "$variant"
  )
  if [[ -n "${EVAL_LIMIT:-}" ]]; then
    args+=(--limit "$EVAL_LIMIT")
  fi
  "$PYTHON" "$LINUX_DIR/run_hf_path_a_eval.py" "${args[@]}" "${extra[@]}"

  local score_args=(--data "$EVAL_SCENARIOS" --predictions "$pred" --no-strict --print-table)
  if [[ -n "${EVAL_LIMIT:-}" ]]; then
    score_args+=(--allow-missing)
  fi
  "$PYTHON" "$ROOT/scripts/workflows/evaluate_mempatch_predictions.py" "${score_args[@]}"

  if [[ "${PATH_A_STRICT_SMOKE:-0}" == "1" ]]; then
    "$PYTHON" - "$pred" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
bad = []
for row in rows:
    audit = row.get("dpa_audit") or {}
    parsed = (audit.get("parse_result") or {}).get("schema_valid")
    rejected = audit.get("rejected_actions") or []
    proposed = audit.get("proposed_actions") or []
    admitted = audit.get("admitted_actions") or []
    if not parsed or rejected or not proposed:
        bad.append((row.get("scenario_id"), parsed, len(proposed), len(admitted), len(rejected)))
if bad:
    raise SystemExit(f"Path A strict smoke failed: {bad}")
PY
  fi
}

if [[ "$VARIANT_FILTER" == "base" || "$VARIANT_FILTER" == "all" ]]; then
  echo "===== Path A base (typed actions + DPA) ====="
  run_variant base --no-adapter
fi

if [[ "$VARIANT_FILTER" == "lora" || "$VARIANT_FILTER" == "all" ]]; then
  echo "===== Path A LoRA (typed actions + DPA) ====="
  run_variant lora_best --adapter-path "$ADAPTER_PATH"
fi

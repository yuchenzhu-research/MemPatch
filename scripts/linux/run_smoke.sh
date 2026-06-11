#!/usr/bin/env bash
# Smoke-test MemPatch (base + LoRA) and all paper baselines with 1 case each.
# Artifacts go to a temp dir and are deleted on success; only smoke.done is kept.
#
#   SLUG=mistral_nemo_12b bash scripts/linux/run_smoke.sh
#   SMOKE_KEEP=1 SLUG=... bash scripts/linux/run_smoke.sh   # keep temp for debugging
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$LINUX_DIR/lib_phases.sh"

SLUG="${SLUG:?set SLUG}"
SMOKE_LIMIT="${SMOKE_LIMIT:-1}"
PERMANENT_RESULT_DIR="$RESULTS_ROOT/$SLUG"
mkdir -p "$PERMANENT_RESULT_DIR" "${LOCAL_ROOT}/tmp"

SMOKE_TMP="$(mktemp -d "${LOCAL_ROOT}/tmp/smoke_${SLUG}_XXXXXX")"
export SMOKE_RESULT_DIR="$SMOKE_TMP"

cleanup_smoke_tmp() {
  if [[ "${SMOKE_KEEP:-0}" == "1" ]]; then
    echo "SMOKE_KEEP=1 — temp artifacts kept at $SMOKE_TMP"
    return 0
  fi
  rm -rf "$SMOKE_TMP"
}
trap cleanup_smoke_tmp EXIT

# Drop legacy smoke files written before temp-dir behavior.
rm -f "$PERMANENT_RESULT_DIR"/smoke.done
rm -f "$PERMANENT_RESULT_DIR"/smoke_* 2>/dev/null || true

log() { echo "[$(date '+%F %T')] [$SLUG] smoke: $*"; }

log "temp dir -> $SMOKE_TMP"

log "MemPatch without LoRA (${SMOKE_LIMIT} case)"
EVAL_PREFIX=smoke_test500 EVAL_LIMIT="$SMOKE_LIMIT" SLUG="$SLUG" \
  bash "$LINUX_DIR/06_eval_test.sh" --variant base

log "MemPatch with LoRA (${SMOKE_LIMIT} case)"
EVAL_PREFIX=smoke_test500 EVAL_LIMIT="$SMOKE_LIMIT" SLUG="$SLUG" \
  bash "$LINUX_DIR/06_eval_test.sh" --variant lora

log "MemPatch Path A with typed actions + DPA (${SMOKE_LIMIT} case)"
PATH_A_STRICT_SMOKE=1 EVAL_PREFIX=smoke_path_a EVAL_LIMIT="$SMOKE_LIMIT" SLUG="$SLUG" \
  bash "$LINUX_DIR/07_eval_path_a.sh" --variant lora

log "paper baseline proxies (${SMOKE_LIMIT} case each)"
BASELINE_SET="${BASELINE_SET:-main}" \
  INCLUDE_LORA=0 \
  EVAL_LIMIT="$SMOKE_LIMIT" \
  PRED_TAG_PREFIX=smoke_baseline_ \
  RESUME=0 \
  SLUG="$SLUG" \
  bash "$LINUX_DIR/run_baseline_matrix.sh"

date -Iseconds >"$PERMANENT_RESULT_DIR/smoke.done"
log "smoke passed -> $PERMANENT_RESULT_DIR/smoke.done (temp artifacts removed on exit)"

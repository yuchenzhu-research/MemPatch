#!/usr/bin/env bash
# Read fixed-split checkpoint selection JSON (subprocess-safe; no export needed).

load_split_index() {
  echo "${SPLIT_INDEX:-0}"
}

load_best_checkpoint() {
  local slug="${1:?slug}"
  local path="${RESULTS_ROOT:?}/${slug}/checkpoint_selection.json"
  [[ -f "$path" ]] || { echo "missing $path" >&2; return 1; }
  "$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p.get("checkpoint_dir") or p.get("best_checkpoint") or p["adapter_dir"])' "$path"
}

selection_matches_run_id() {
  local selection="$1"
  local run_id="$2"
  "$PYTHON" - "$selection" "$run_id" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
run_id = sys.argv[2]
paths = (
    str(payload.get("checkpoint_dir", "")),
    str(payload.get("best_checkpoint", "")),
    str(payload.get("adapter_dir", "")),
    str(payload.get("run_id", "")),
    str(payload.get("log_dir", "")),
)
raise SystemExit(0 if any(run_id in p for p in paths if p) else 1)
PY
}

ensure_selection() {
  local slug="${1:?slug}"
  local result_dir="${RESULTS_ROOT:?}/${slug}"
  local selection="${result_dir}/checkpoint_selection.json"
  local valid=0
  if [[ -f "$selection" ]] && selection_matches_run_id "$selection" "$RUN_ID"; then
    valid=1
  fi
  if [[ "$valid" != "1" ]]; then
    if [[ -f "$LINUX_DIR/discover_checkpoint.py" ]] \
      && "$PYTHON" "$LINUX_DIR/discover_checkpoint.py" \
        --slug "$slug" \
        --log-root "$LOG_ROOT" \
        --adapter-root "$ADAPTER_ROOT" \
        --out "$selection" \
        --prefer-run-id "$RUN_ID" >/dev/null 2>&1; then
      :
    elif SLUG="$slug" bash "${LINUX_DIR:?}/05_pick_best.sh" >/dev/null; then
      :
    else
      echo "error: no LoRA checkpoint for $slug (train or set BEST_CHECKPOINT=...)" >&2
      return 1
    fi
  fi
  load_best_checkpoint "$slug"
}

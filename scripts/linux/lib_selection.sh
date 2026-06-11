#!/usr/bin/env bash
# Read k-fold / checkpoint selection JSON (subprocess-safe; no export needed).

load_best_fold() {
  echo "${VALIDATION_PART:-0}"
}

load_best_checkpoint() {
  local slug="${1:?slug}"
  local path="${RESULTS_ROOT:?}/${slug}/checkpoint_selection.json"
  [[ -f "$path" ]] || { echo "missing $path" >&2; return 1; }
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_dir"])' "$path"
}

ensure_selection() {
  local slug="${1:?slug}"
  local result_dir="${RESULTS_ROOT:?}/${slug}"
  local selection="${result_dir}/checkpoint_selection.json"
  local valid=0
  if [[ -f "$selection" ]] && "$PYTHON" - "$selection" "$RUN_ID" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
raise SystemExit(0 if sys.argv[2] in str(payload.get("checkpoint_dir", "")) else 1)
PY
  then
    valid=1
  fi
  if [[ "$valid" != "1" ]]; then
    SLUG="$slug" bash "${LINUX_DIR:?}/05_pick_best.sh" >/dev/null
  fi
  load_best_checkpoint "$slug"
}

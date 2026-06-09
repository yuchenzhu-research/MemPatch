#!/usr/bin/env bash
# Read k-fold / checkpoint selection JSON (subprocess-safe; no export needed).

load_best_fold() {
  local slug="${1:?slug}"
  local path="${RESULTS_ROOT:?}/${slug}/kfold_selection.json"
  [[ -f "$path" ]] || { echo "missing $path" >&2; return 1; }
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_fold"])' "$path"
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
  if [[ ! -f "${result_dir}/checkpoint_selection.json" ]]; then
    SLUG="$slug" bash "${LINUX_DIR:?}/05_pick_best.sh" >/dev/null
  fi
  load_best_checkpoint "$slug"
}

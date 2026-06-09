#!/usr/bin/env bash
# Run RAG + Full-context external memory baselines on four MLX models, then score.
#
# Each run: scenarios → build_prompt (JSON schema + enums) → MLX base → predictions.jsonl
#           → benchmark.api.evaluate_predictions → *_metrics.json
#
# Manual usage (from repo root):
#
#   # 1) ensure MLX env
#   python3.12 -m venv .venv && .venv/bin/pip install -e ".[mlx]"
#   .venv/bin/python -c "import mlx.core, mlx_lm; print('ok')"
#
#   # 2) run all 4 models × RAG + Full (default: test split, first 100 cases)
#   bash scripts/workflows/run_external_memory_baselines.sh
#
#   # 3) full test500
#   LIMIT=500 bash scripts/workflows/run_external_memory_baselines.sh
#
#   # 4) force re-run even if metrics exist
#   FORCE=1 bash scripts/workflows/run_external_memory_baselines.sh
#
#   # 5) single model / backend
#   MODELS=qwen3_14b BACKENDS=rag bash scripts/workflows/run_external_memory_baselines.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-$ROOT:$ROOT/src:$ROOT/scripts}"

DATA="${DATA:-$ROOT/hf_release/mempatch/test/scenarios.jsonl}"
OUT="${OUT:-$ROOT/local/runs/baselines/external_memory}"
LIMIT="${LIMIT:-100}"
OFFSET="${OFFSET:-0}"
FORCE="${FORCE:-0}"
BACKENDS="${BACKENDS:-rag,full}"
MODELS="${MODELS:-qwen3_14b,gemma3_12b,mistral_nemo_12b,llama3_1_8b}"

mkdir -p "$OUT"
LOG="$OUT/run.log"

log() { printf '[baselines] %s\n' "$*" | tee -a "$LOG"; }

model_dir_for() {
  case "$1" in
    qwen3_14b) echo "$ROOT/local/models/Qwen3-14B-MLX-4bit" ;;
    gemma3_12b) echo "$ROOT/local/models/gemma-3-12b-it-4bit" ;;
    mistral_nemo_12b) echo "$ROOT/local/models/Mistral-Nemo-Instruct-2407-4bit" ;;
    llama3_1_8b) echo "$ROOT/local/models/Meta-Llama-3.1-8B-Instruct-4bit" ;;
    *) echo "unknown model slug: $1" >&2; exit 1 ;;
  esac
}

require_mlx() {
  if ! "$PYTHON" -c "
from scripts._root import bootstrap_from
bootstrap_from('scripts/eval/run_mempatch_memory_baselines.py')
import mlx.core as mx
import mlx_lm
print(mx.__version__)
" 2>/dev/null; then
    echo "error: MLX not available in $PYTHON (after scripts bootstrap)" >&2
    echo "fix: python3.12 -m venv .venv && .venv/bin/pip install -e \".[mlx]\"" >&2
    echo "note: scripts/mlx/ was renamed to scripts/mlx_support/ to avoid shadowing PyPI mlx" >&2
    exit 1
  fi
}

run_one() {
  local slug="$1" backend="$2"
  local model_dir
  model_dir="$(model_dir_for "$slug")"
  if [[ ! -d "$model_dir" ]]; then
    log "ERROR missing model dir: $model_dir (download first)"
    return 1
  fi

  local split_tag="test${LIMIT}"
  if [[ "$LIMIT" == "500" ]]; then split_tag="test500"; fi
  local tag="${slug}_${backend}_${split_tag}"
  local pred="$OUT/${tag}_predictions.jsonl"
  local metrics="$OUT/${tag}_metrics.json"

  if [[ "$FORCE" != "1" && -f "$metrics" ]]; then
    log "skip $tag (metrics exist; set FORCE=1 to rerun)"
    return 0
  fi

  log "RUN $tag | backend=$backend model=$model_dir limit=$LIMIT offset=$OFFSET"
  "$PYTHON" "$ROOT/scripts/eval/run_mempatch_memory_baselines.py" \
    --data "$DATA" \
    --backend "$backend" \
    --limit "$LIMIT" \
    --offset "$OFFSET" \
    --model "$model_dir" \
    --out-predictions "$pred" \
    --out-metrics "$metrics"
}

print_summary() {
  log "summary (headline metrics):"
  "$PYTHON" - <<PY
import json
import os
from pathlib import Path

out = Path("${OUT}")
rows = []
for p in sorted(out.glob("*_metrics.json")):
    d = json.loads(p.read_text())
    h = d.get("headline_metrics") or {}
    rows.append(
        (
            p.name.replace("_metrics.json", ""),
            d.get("backend", "?"),
            d.get("count", "?"),
            h.get("joint_revision_success"),
            h.get("decision_macro_f1"),
            h.get("memory_state_accuracy"),
            h.get("evidence_f1"),
        )
    )
if not rows:
    print("(no metrics files yet)")
else:
    print(f"{'run':36} {'backend':6} {'n':>4} {'joint':>8} {'dec_f1':>8} {'mem':>8} {'ev_f1':>8}")
    for r in rows:
        print(f"{r[0]:36} {str(r[1]):6} {str(r[2]):>4} {str(r[3]):>8} {str(r[4]):>8} {str(r[5]):>8} {str(r[6]):>8}")
PY
}

main() {
  require_mlx
  log "data=$DATA out=$OUT limit=$LIMIT offset=$OFFSET backends=$BACKENDS models=$MODELS"

  IFS=',' read -r -a backend_list <<< "$BACKENDS"
  IFS=',' read -r -a model_list <<< "$MODELS"

  for slug in "${model_list[@]}"; do
    slug="${slug// /}"
    [[ -z "$slug" ]] && continue
    for backend in "${backend_list[@]}"; do
      backend="${backend// /}"
      [[ -z "$backend" ]] && continue
      run_one "$slug" "$backend"
    done
  done

  print_summary
  log "done"
}

main "$@"

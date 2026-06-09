#!/usr/bin/env bash
# Smoke 50: RAG + Full baselines (base MLX) and Path B LoRA for all four paper models.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-$ROOT:$ROOT/src:$ROOT/scripts}"
OUT="${OUT:-$ROOT/local/runs/baselines/smoke50}"
DATA="${DATA:-$ROOT/hf_release/mempatch/test/scenarios.jsonl}"
SFT="${SFT:-$ROOT/local/train_data/paper/test500/sft.jsonl}"
LIMIT="${LIMIT:-50}"

mkdir -p "$OUT"
LOG="$OUT/matrix.log"
exec > >(tee -a "$LOG") 2>&1

log() { printf '[smoke50] %s\n' "$*"; }

run_baseline() {
  local slug="$1" model_dir="$2" backend="$3"
  local tag="${slug}_${backend}_test50"
  if [[ -f "$OUT/${tag}_metrics.json" ]]; then
    log "skip baseline $tag (metrics exist)"
    return 0
  fi
  log "baseline $tag"
  "$PYTHON" "$ROOT/scripts/eval/run_mempatch_memory_baselines.py" \
    --data "$DATA" --backend "$backend" --limit "$LIMIT" \
    --model "$model_dir" \
    --out-predictions "$OUT/${tag}.jsonl" \
    --out-metrics "$OUT/${tag}_metrics.json"
}

run_lora() {
  local slug="$1" model_dir="$2" adapter="$3"
  local tag="${slug}_pathB_lora_test50"
  if [[ -f "$OUT/${tag}_metrics.json" ]]; then
    log "skip lora $tag (metrics exist)"
    return 0
  fi
  log "path B $tag"
  "$PYTHON" "$ROOT/scripts/eval/run_mlx_lora_smoke_eval.py" \
    --data "$SFT" --eval-data "$DATA" --limit "$LIMIT" \
    --model "$model_dir" --adapter-path "$adapter" \
    --out-predictions "$OUT/${tag}.jsonl" \
    --out-metrics "$OUT/${tag}_metrics.json"
}

MODELS=(
  "qwen3_14b|$ROOT/local/models/Qwen3-14B-MLX-4bit|$ROOT/local/adapters/qwen3_14b_pathB_lora"
  "gemma3_12b|$ROOT/local/models/gemma-3-12b-it-4bit|$ROOT/local/adapters/gemma3_12b_pathB_lora"
  "mistral_nemo_12b|$ROOT/local/models/Mistral-Nemo-Instruct-2407-4bit|$ROOT/local/adapters/mistral_nemo_12b_pathB_lora"
  "llama3_1_8b|$ROOT/local/models/Meta-Llama-3.1-8B-Instruct-4bit|$ROOT/local/adapters/llama3_1_8b_pathB_lora"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r slug model adapter <<< "$entry"
  for backend in rag full; do
    run_baseline "$slug" "$model" "$backend"
  done
  run_lora "$slug" "$model" "$adapter"
done

log "done — summary:"
"$PYTHON" - <<'PY'
import json
from pathlib import Path
out = Path(__import__("os").environ.get("OUT", "local/runs/baselines/smoke50"))
rows = []
for p in sorted(out.glob("*_metrics.json")):
    d = json.loads(p.read_text())
    h = d.get("headline_metrics") or d
    rows.append((p.stem, h.get("joint_revision_success"), h.get("decision_macro_f1")))
print(f"{'run':40} {'joint':>8} {'dec_f1':>8}")
for name, j, d in rows:
    print(f"{name:40} {j!s:>8} {d!s:>8}")
PY

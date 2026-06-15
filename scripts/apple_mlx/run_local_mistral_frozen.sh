#!/usr/bin/env bash
# Frozen test500 campaign for local Mistral MLX. No training/LoRA.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MODEL="${MODEL:-$ROOT/local/models/Mistral-Nemo-Instruct-2407-4bit}"
DATA="${DATA:-$ROOT/supplement/data/test/scenarios.jsonl}"
RESULTS="${RESULTS:-$ROOT/local/results/final/mistral_nemo_12b}"
LIMIT="${LIMIT:-}"
LIMIT_ARGS=()
[[ -n "$LIMIT" ]] && LIMIT_ARGS=(--limit "$LIMIT")

[[ -d "$MODEL" ]] || { echo "error: missing MLX model: $MODEL" >&2; exit 1; }
[[ -f "$DATA" ]] || { echo "error: missing test500: $DATA" >&2; exit 1; }
mkdir -p "$RESULTS" "$ROOT/local/logs"

for baseline in structured_direct full_context vanilla_rag time_aware_rag summary_memory; do
  "$PYTHON" -m scripts.apple_mlx.run_mlx_baselines \
    --baseline "$baseline" --eval-data "$DATA" --model "$MODEL" \
    --results-dir "$RESULTS" --model-tag mistral_nemo_12b "${LIMIT_ARGS[@]}"
done

"$PYTHON" "$ROOT/scripts/linux/aggregate_baseline_table.py" \
  --results-dir "$RESULTS" --out "$RESULTS/baseline_matrix.md"

"$PYTHON" -m scripts.apple_mlx.run_mlx_path_a_eval \
  --eval-data "$DATA" --model "$MODEL" --results-dir "$RESULTS" \
  --model-tag mistral_nemo_12b "${LIMIT_ARGS[@]}"

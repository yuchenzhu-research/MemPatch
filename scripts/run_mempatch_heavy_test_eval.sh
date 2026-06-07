#!/usr/bin/env bash
# Held-out test split eval for the heavy MLX LoRA adapter (do not git add local/).
#
# Usage (from repo root):
#   chmod +x scripts/run_mempatch_heavy_test_eval.sh
#   ./scripts/run_mempatch_heavy_test_eval.sh           # print help
#   ./scripts/run_mempatch_heavy_test_eval.sh prepare   # build test500_sft.jsonl only
#   ./scripts/run_mempatch_heavy_test_eval.sh smoke     # quick 50-case LoRA eval
#   ./scripts/run_mempatch_heavy_test_eval.sh full      # base + LoRA on all 500 (~1-2h)
#   ./scripts/run_mempatch_heavy_test_eval.sh analyze   # error analysis + compare (needs predictions)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH=.:src

DATA_DIR="local/train_data/mempatch_v13_heavy"
ADAPTER_DIR="local/adapters/qwen3_14b_mempatch_v13_heavy"
TEST_SFT="$DATA_DIR/test500_sft.jsonl"
SCENARIOS="hf_release/mempatch/test/scenarios.jsonl"

BASE_PRED="local/results/qwen3_14b_base_heavy_test500_predictions.jsonl"
LORA_PRED="local/results/qwen3_14b_heavy_test500_predictions.jsonl"
BASE_METRICS="local/results/qwen3_14b_base_heavy_test500_metrics.json"
LORA_METRICS="local/results/qwen3_14b_heavy_test500_metrics.json"
ERROR_JSON="local/results/qwen3_14b_heavy_test500_error_analysis.json"

step_prepare() {
  echo "== Prepare test500 SFT inference rows =="
  .venv/bin/python - <<'PY'
from pathlib import Path
from scripts.prepare_mempatch_v13_smoke import read_jsonl, sft_example, write_jsonl

root = Path(".")
out_dir = root / "local/train_data/mempatch_v13_heavy"
out_dir.mkdir(parents=True, exist_ok=True)
test = read_jsonl(root / "hf_release/mempatch/test/scenarios.jsonl")
out = out_dir / "test500_sft.jsonl"
write_jsonl(out, [sft_example(s) for s in test])
print(f"wrote {len(test)} rows -> {out}")
PY
}

step_smoke() {
  step_prepare
  echo "== Smoke eval: heavy LoRA on first 50 test cases =="
  .venv/bin/python scripts/run_mlx_lora_smoke_eval.py \
    --data "$TEST_SFT" \
    --eval-data "$SCENARIOS" \
    --adapter-path "$ADAPTER_DIR" \
    --limit 50 \
    --out-predictions local/results/qwen3_14b_heavy_test50_predictions.jsonl \
    --out-metrics local/results/qwen3_14b_heavy_test50_metrics.json
  .venv/bin/python scripts/evaluate_mempatch_predictions.py \
    --data "$SCENARIOS" \
    --predictions local/results/qwen3_14b_heavy_test50_predictions.jsonl \
    --no-strict --allow-missing --print-table
}

step_full() {
  step_prepare
  echo "== Full test500: base (no adapter) =="
  .venv/bin/python scripts/run_mlx_lora_smoke_eval.py \
    --data "$TEST_SFT" \
    --eval-data "$SCENARIOS" \
    --no-adapter \
    --out-predictions "$BASE_PRED" \
    --out-metrics "$BASE_METRICS"

  echo "== Full test500: heavy LoRA =="
  .venv/bin/python scripts/run_mlx_lora_smoke_eval.py \
    --data "$TEST_SFT" \
    --eval-data "$SCENARIOS" \
    --adapter-path "$ADAPTER_DIR" \
    --out-predictions "$LORA_PRED" \
    --out-metrics "$LORA_METRICS"

  step_analyze
}

step_analyze() {
  echo "== Error analysis (base vs heavy LoRA on test500) =="
  .venv/bin/python scripts/analyze_mlx_lora_errors.py \
    --data "$SCENARIOS" \
    --base-predictions "$BASE_PRED" \
    --lora-predictions "$LORA_PRED" \
    --out-json "$ERROR_JSON" \
    --show-cases 8

  echo "== Headline table (heavy LoRA) =="
  .venv/bin/python scripts/evaluate_mempatch_predictions.py \
    --data "$SCENARIOS" \
    --predictions "$LORA_PRED" \
    --no-strict --allow-missing --print-table

  echo "== Compare v2 smoke / heavy valid200 / heavy test500 =="
  .venv/bin/python - <<'PY'
import json
from pathlib import Path

keys = [
    "decision_macro_f1",
    "memory_state_accuracy",
    "evidence_f1",
    "minimal_evidence_exact_match",
    "failure_diagnosis_accuracy",
    "joint_revision_success",
    "answer_state_consistency",
]
runs = {
    "v2_lora_valid100":   "local/results/qwen3_14b_lora_v2_valid_metrics.json",
    "heavy_lora_valid200": "local/results/qwen3_14b_heavy_valid_metrics.json",
    "heavy_lora_test500":  "local/results/qwen3_14b_heavy_test500_metrics.json",
}
for label, path in runs.items():
    p = Path(path)
    if not p.exists():
        print(f"\n{label}: (not found)")
        continue
    payload = json.loads(p.read_text())
    m = payload["headline_metrics"]
    print(f"\n{label} (n={payload['count']}):")
    for k in keys:
        v = m.get(k)
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
PY
}

print_instructions() {
  cat <<EOF
MemPatch heavy adapter — held-out TEST eval

  export PYTHONPATH=.:src

  ./scripts/run_mempatch_heavy_test_eval.sh prepare   # test500_sft.jsonl only
  ./scripts/run_mempatch_heavy_test_eval.sh smoke     # 50-case quick check (~10 min)
  ./scripts/run_mempatch_heavy_test_eval.sh full      # base + LoRA on 500 (~1-2 h)
  ./scripts/run_mempatch_heavy_test_eval.sh analyze   # re-run analysis if predictions exist

Requires:
  local/adapters/qwen3_14b_mempatch_v13_heavy/adapters.safetensors
  local/models/Qwen3-14B-MLX-4bit/

Outputs:
  $LORA_METRICS
  $ERROR_JSON
EOF
}

case "${1:-}" in
  prepare) step_prepare ;;
  smoke)   step_smoke ;;
  full)    step_full ;;
  analyze) step_analyze ;;
  ""|help|-h|--help) print_instructions ;;
  *)
    echo "Unknown step: $1" >&2
    print_instructions
    exit 1
    ;;
esac

#!/usr/bin/env bash
# Deterministic 30-case no-LoRA smoke: frozen direct, full context, MemPatch zero-shot.
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$LINUX_DIR/lib_gpu.sh"

DRY_RUN="${DRY_RUN:-0}"
SMOKE_ROOT="$ROOT/artifacts/smoke"
SMOKE_SCENARIOS="$SMOKE_ROOT/scenarios.jsonl"
SMOKE_SFT="$SMOKE_ROOT/eval_bundle/sft.jsonl"
PRED_ROOT="$ROOT/predictions/smoke_no_lora"
SCORE_ROOT="$ROOT/scores/smoke_no_lora"
BUNDLE_ROOT="$SMOKE_ROOT/run_bundles"

run() {
  printf '+ '; printf '%q ' "$@"; printf '\n'
  [[ "$DRY_RUN" == "1" ]] || "$@"
}

if [[ "$DRY_RUN" == "1" ]]; then
  TEST_SCENARIOS="${TEST_SCENARIOS:-$LOCAL_ROOT/data/mempatch/test/scenarios.jsonl}"
else
  resolve_test_scenarios
fi
run "$PYTHON" "$ROOT/scripts/select_smoke_cases.py" \
  --input "$TEST_SCENARIOS" \
  --ids-out "$SMOKE_ROOT/smoke_case_ids.json" \
  --scenarios-out "$SMOKE_SCENARIOS"
run "$PYTHON" "$ROOT/scripts/data/build_paper_eval_bundle.py" \
  --scenarios "$SMOKE_SCENARIOS" --out-dir "$SMOKE_ROOT/eval_bundle"

for slug in "${SMOKE_SLUGS[@]}"; do
  artifact_id="$(artifact_model_id "$slug")"
  model="$(resolve_hf_model "$slug")"
  pred_dir="$PRED_ROOT/$artifact_id"
  score_dir="$SCORE_ROOT/$artifact_id"
  bundle_dir="$BUNDLE_ROOT/$artifact_id"
  run mkdir -p "$pred_dir" "$score_dir" "$bundle_dir"

  for spec in \
    "structured_direct:frozen_direct_prompting" \
    "full_context:full_context"; do
    baseline="${spec%%:*}"
    system="${spec##*:}"
    run "$PYTHON" "$LINUX_DIR/run_hf_baselines.py" \
      --baseline "$baseline" --eval-data "$SMOKE_SCENARIOS" --model-id "$model" \
      --out-predictions "$bundle_dir/${system}.jsonl" \
      --out-metrics "$bundle_dir/${system}_metrics.json" \
      --split-tag "$system" --model-tag "$artifact_id"
    run cp "$bundle_dir/${system}.jsonl" "$pred_dir/${system}.jsonl"
    run cp "$bundle_dir/${system}_scored.jsonl" "$score_dir/${system}_scored.jsonl"
  done

  system="mempatch_zero_shot"
  run "$PYTHON" "$LINUX_DIR/run_hf_path_a_eval.py" \
    --data "$SMOKE_SFT" --eval-data "$SMOKE_SCENARIOS" --model-id "$model" \
    --no-adapter --out-predictions "$bundle_dir/${system}.jsonl" \
    --split-tag "$system" --model-tag "$artifact_id" --variant-tag zero_shot
  run cp "$bundle_dir/${system}.jsonl" "$pred_dir/${system}.jsonl"
  run cp "$bundle_dir/${system}_scored.jsonl" "$score_dir/${system}_scored.jsonl"
  release_gpu
done

run "$PYTHON" "$ROOT/scripts/paper/build_experiment_artifacts.py" smoke \
  --bundle-root "$BUNDLE_ROOT" --out-root "$ROOT"

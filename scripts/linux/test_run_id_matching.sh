#!/usr/bin/env bash
# Test harness to verify exact RUN_ID path segment checking.
# Prevents substring contamination (e.g., full512 vs full512_2048).
set -euo pipefail

LINUX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$LINUX_DIR/env.sh"
source "$LINUX_DIR/lib_phases.sh"

mkdir -p "${LOCAL_ROOT}/tmp"
TEST_TMP="$(mktemp -d "${LOCAL_ROOT}/tmp/test_matching_XXXXXX")"
cleanup() {
  rm -rf "$TEST_TMP"
}
trap cleanup EXIT

export RESULTS_ROOT="$TEST_TMP"

SLUG="phi4_14b"
SLUG_DIR="$RESULTS_ROOT/$SLUG"
mkdir -p "$SLUG_DIR"

# 1. Test phase_pick_done
echo '{"checkpoint_dir": "/adapters/phi4_14b_multitask_lora/split0/full512/checkpoint-128"}' > "$SLUG_DIR/checkpoint_selection.json"

echo "=== Testing phase_pick_done exact matching ==="
RUN_ID=full512 phase_pick_done "$SLUG" && echo "PASS: RUN_ID=full512 matches full512" || { echo "FAIL: RUN_ID=full512 should match"; exit 1; }
! RUN_ID=full512_2048 phase_pick_done "$SLUG" && echo "PASS: RUN_ID=full512_2048 does not match full512" || { echo "FAIL: RUN_ID=full512_2048 matched (substring error!)"; exit 1; }
! RUN_ID=phi4_smoke10 phase_pick_done "$SLUG" && echo "PASS: RUN_ID=phi4_smoke10 does not match full512" || { echo "FAIL: RUN_ID=phi4_smoke10 matched"; exit 1; }

# 2. Test phase_eval_done
cat <<EOF > "$SLUG_DIR/test500_lora_best_manifest.json"
{
  "run_meta": {
    "adapter_path": "/adapters/phi4_14b_multitask_lora/split0/full512/checkpoint-128",
    "schema_projection": "public_only_v1"
  }
}
EOF

cat <<EOF > "$SLUG_DIR/test500_path_a_lora_best_manifest.json"
{
  "run_meta": {
    "adapter_path": "/adapters/phi4_14b_multitask_lora/split0/full512/checkpoint-128",
    "method_path": "path_a_typed_actions_dpa"
  }
}
EOF

cat <<EOF > "$SLUG_DIR/test500_path_a_lora_best_no_dpa_manifest.json"
{
  "run_meta": {
    "adapter_path": "/adapters/phi4_14b_multitask_lora/split0/full512/checkpoint-128",
    "method_path": "path_a_typed_actions_no_dpa"
  }
}
EOF

touch "$SLUG_DIR/test500_base_predictions.jsonl"
touch "$SLUG_DIR/test500_lora_best_predictions.jsonl"
touch "$SLUG_DIR/test500_path_a_lora_best_predictions.jsonl"

echo "=== Testing phase_eval_done exact matching ==="
RUN_ID=full512 phase_eval_done "$SLUG" && echo "PASS: RUN_ID=full512 eval matches full512" || { echo "FAIL: RUN_ID=full512 eval should match"; exit 1; }
! RUN_ID=full512_2048 phase_eval_done "$SLUG" && echo "PASS: RUN_ID=full512_2048 eval does not match full512" || { echo "FAIL: RUN_ID=full512_2048 eval matched"; exit 1; }
! RUN_ID=phi4_smoke10 phase_eval_done "$SLUG" && echo "PASS: RUN_ID=phi4_smoke10 eval does not match full512" || { echo "FAIL: RUN_ID=phi4_smoke10 eval matched"; exit 1; }

echo "ALL RUN_ID MATCHING TESTS PASSED!"
exit 0

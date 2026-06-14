#!/usr/bin/env bash
# Pipeline phase detection for one SLUG (sources env.sh first).

phase_prefetch_done() {
  local slug="${1:?slug}"
  local hub_id local_dir
  hub_id="$(resolve_hf_model_hub "$slug")" || return 1
  local_dir="$(local_model_dir_for_hub "$hub_id")"
  model_dir_complete "$local_dir"
}

phase_train_done() {
  local slug="${1:?slug}"
  [[ -f "$LOG_ROOT/${slug}_split${SPLIT_INDEX}/${RUN_ID}/trainer_metrics.json" ]] || return 1
  local ckpt_count
  ckpt_count="$(find "$ADAPTER_ROOT/${slug}_multitask_lora/split${SPLIT_INDEX}/${RUN_ID}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | wc -l)"
  [[ "$ckpt_count" -ge 1 ]]
}

phase_pick_done() {
  local slug="${1:?slug}"
  local selection="$RESULTS_ROOT/$slug/checkpoint_selection.json"
  [[ -f "$selection" ]] || return 1
  "$PYTHON" - "$selection" "$RUN_ID" <<'PY'
import json, sys
from pathlib import Path
payload = json.load(open(sys.argv[1]))
checkpoint = Path(str(payload.get("checkpoint_dir", "")))
raise SystemExit(0 if checkpoint.parent.name == sys.argv[2] else 1)
PY
}

phase_eval_done() {
  local slug="${1:?slug}"
  [[ -f "$RESULTS_ROOT/$slug/test500_base_predictions.jsonl" ]] \
    && [[ -f "$RESULTS_ROOT/$slug/test500_lora_best_predictions.jsonl" ]] \
    && [[ -f "$RESULTS_ROOT/$slug/test500_lora_best_manifest.json" ]] \
    && [[ -f "$RESULTS_ROOT/$slug/test500_path_a_lora_best_predictions.jsonl" ]] \
    && [[ -f "$RESULTS_ROOT/$slug/test500_path_a_lora_best_manifest.json" ]] \
    && [[ -f "$RESULTS_ROOT/$slug/test500_path_a_lora_best_no_dpa_manifest.json" ]] || return 1
  "$PYTHON" - \
    "$RESULTS_ROOT/$slug/test500_lora_best_manifest.json" \
    "$RESULTS_ROOT/$slug/test500_path_a_lora_best_manifest.json" \
    "$RESULTS_ROOT/$slug/test500_path_a_lora_best_no_dpa_manifest.json" \
    "$RUN_ID" <<'PY'
import json, sys
from pathlib import Path

def meta(path):
    return (json.load(open(path)).get("run_meta") or {})

def run_id(payload):
    adapter = payload.get("adapter_path")
    return Path(str(adapter)).parent.name if adapter else None

path_b_meta = meta(sys.argv[1])
path_a_meta = meta(sys.argv[2])
no_dpa_meta = meta(sys.argv[3])
expected_run_id = sys.argv[4]
ok = run_id(path_b_meta) == expected_run_id
ok = ok and run_id(path_a_meta) == expected_run_id
ok = ok and run_id(no_dpa_meta) == expected_run_id
ok = ok and path_b_meta.get("schema_projection") == "public_only_v1"
ok = ok and path_a_meta.get("method_path") == "path_a_typed_actions_dpa"
ok = ok and no_dpa_meta.get("method_path") == "path_a_typed_actions_no_dpa"
raise SystemExit(0 if ok else 1)
PY
}

phase_smoke_done() {
  local slug="${1:?slug}"
  [[ -f "$RESULTS_ROOT/$slug/smoke.done" ]]
}

phase_baselines_done() {
  local slug="${1:?slug}"
  [[ -f "$RESULTS_ROOT/$slug/baselines_full.done" ]]
}

print_model_status() {
  local slug="${1:?slug}"
  require_paper_slug "$slug"
  local hub_id
  hub_id="$(resolve_hf_model "$slug")" || return 1
  echo "slug=$slug model=$hub_id"
  phase_prefetch_done "$slug" && echo "  prefetch: OK" || echo "  prefetch: MISSING"
  phase_train_done "$slug" && echo "  train (single split, $TRAIN_ITERS steps): OK" || echo "  train: incomplete"
  phase_pick_done "$slug" && echo "  pick (checkpoint): OK" || echo "  pick: MISSING"
  phase_eval_done "$slug" && echo "  Path A DPA + Path B direct (500): OK" || echo "  Path A/B eval: incomplete"
  phase_baselines_done "$slug" && echo "  public baselines: OK" || echo "  baselines: incomplete"
}

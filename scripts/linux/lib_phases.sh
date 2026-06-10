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
  local fold
  for fold in $(seq 0 $((KFOLDS - 1))); do
    [[ -f "$LOG_ROOT/${slug}_fold${fold}/trainer_metrics.json" ]] || return 1
    local ckpt_count
    ckpt_count="$(find "$ADAPTER_ROOT/${slug}_pathB_lora/fold${fold}/${RUN_ID}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | wc -l)"
    [[ "$ckpt_count" -ge 1 ]] || return 1
  done
}

phase_pick_done() {
  local slug="${1:?slug}"
  [[ -f "$RESULTS_ROOT/$slug/kfold_selection.json" ]] \
    && [[ -f "$RESULTS_ROOT/$slug/checkpoint_selection.json" ]]
}

phase_eval_done() {
  local slug="${1:?slug}"
  [[ -f "$RESULTS_ROOT/$slug/test500_base_predictions.jsonl" ]] \
    && [[ -f "$RESULTS_ROOT/$slug/test500_lora_best_predictions.jsonl" ]]
}

phase_baselines_done() {
  local slug="${1:?slug}"
  [[ -f "$RESULTS_ROOT/$slug/mempatch_lora_best_predictions.jsonl" ]]
}

print_model_status() {
  local slug="${1:?slug}"
  require_paper_slug "$slug"
  local hub_id
  hub_id="$(resolve_hf_model "$slug")" || return 1
  echo "slug=$slug model=$hub_id"
  phase_prefetch_done "$slug" && echo "  prefetch: OK" || echo "  prefetch: MISSING"
  phase_train_done "$slug" && echo "  train (5-fold): OK" || echo "  train (5-fold): incomplete"
  phase_pick_done "$slug" && echo "  pick (fold+ckpt): OK" || echo "  pick: MISSING"
  phase_eval_done "$slug" && echo "  eval (with/without): OK" || echo "  eval: incomplete"
  phase_baselines_done "$slug" && echo "  baselines (11+1): OK" || echo "  baselines: incomplete"
}

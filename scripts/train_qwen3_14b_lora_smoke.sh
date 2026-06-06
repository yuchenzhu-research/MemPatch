#!/usr/bin/env bash
# MemPatch Qwen3-14B 4-bit LoRA smoke training (manual run only).
#
# Prerequisites:
#   pip install -U "mlx-lm[train]"
#
# Data prep (from repo root):
#   python scripts/prepare_mempatch_sft.py \
#     --main local/MemPatch/main/scenarios.jsonl \
#     --hard local/MemPatch/hard/scenarios.jsonl \
#     --out-dir local/train_data/mempatch_qwen14b_smoke \
#     --train-size 512 --valid-size 64 --hard-probe-size 50
#
# Config reference: experiments/configs/qwen3_14b_lora_smoke.yaml
#
# NOTE: mlx-lm QLoRA requires a pre-quantized MLX model checkpoint.
# The MemPatch config names Qwen/Qwen3-14B-Instruct; for 4-bit training use
# an MLX 4-bit variant such as Qwen/Qwen3-14B-MLX-4bit (downloads on first run).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT}/experiments/configs/qwen3_14b_lora_smoke.yaml"
DATA_DIR="${ROOT}/local/train_data/mempatch_qwen14b_smoke"
ADAPTER_DIR="${ROOT}/local/adapters/qwen3_14b_mempatch_smoke"
LOG_DIR="${ROOT}/local/logs/qwen3_14b_mempatch_smoke"
MLX_CONFIG="${LOG_DIR}/mlx_lora.yaml"

# Override with: OOM=1 bash scripts/train_qwen3_14b_lora_smoke.sh
OOM="${OOM:-0}"

if [[ "${OOM}" == "1" ]]; then
  MAX_SEQ_LEN=1536
  ITERS=32
  LORA_RANK=4
  LORA_KEYS='["self_attn.q_proj", "self_attn.v_proj"]'
else
  MAX_SEQ_LEN=2048
  ITERS=64
  LORA_RANK=8
  LORA_KEYS='["self_attn.q_proj", "self_attn.v_proj", "self_attn.o_proj"]'
fi

# QLoRA base weights (4-bit MLX). Change only if you have a local converted copy.
MLX_MODEL="${MLX_MODEL:-Qwen/Qwen3-14B-MLX-4bit}"

mkdir -p "${LOG_DIR}" "${ADAPTER_DIR}"

cat > "${MLX_CONFIG}" <<EOF
model: "${MLX_MODEL}"
train: true
fine_tune_type: lora
optimizer: adamw
data: "${DATA_DIR}"
seed: 2027
batch_size: 1
iters: ${ITERS}
learning_rate: 1.0e-5
max_seq_length: ${MAX_SEQ_LEN}
grad_accumulation_steps: 8
grad_checkpoint: true
mask_prompt: true
adapter_path: "${ADAPTER_DIR}"
save_every: 32
steps_per_eval: 32
val_batches: -1
lora_parameters:
  keys: ${LORA_KEYS}
  rank: ${LORA_RANK}
  scale: 16.0
  dropout: 0.05
EOF

echo "MLX LoRA config written to ${MLX_CONFIG}"
echo "Model: ${MLX_MODEL} | iters=${ITERS} | max_seq_length=${MAX_SEQ_LEN} | lora_rank=${LORA_RANK}"

# Primary entrypoint (mlx-lm >= 0.25): "${PYTHON:-python}" -m mlx_lm lora
"${PYTHON:-python}" -m mlx_lm lora --config "${MLX_CONFIG}"

# ---------------------------------------------------------------------------
# Post-training hard_probe_50 evaluation (manual)
# ---------------------------------------------------------------------------
# TODO: add scripts/run_mempatch_mlx.py to batch-generate benchmark predictions
# from hard_probe.jsonl with/without --adapter-path. model_runner.py only
# supports remote API providers today.
#
# Direct baseline (no adapter):
#   python -m mlx_lm generate \
#     --model "${MLX_MODEL}" \
#     --max-tokens 1024 \
#     --prompt '<build_prompt(public_view) JSON>'
#
# LoRA prediction:
#   python -m mlx_lm generate \
#     --model "${MLX_MODEL}" \
#     --adapter-path "${ADAPTER_DIR}" \
#     --max-tokens 1024 \
#     --prompt '<same prompt>'
#
# Score predictions (requires full hard scenarios with hidden_gold):
#   PYTHONPATH=. python scripts/evaluate_mempatch_predictions.py \
#     --data local/MemPatch/hard/scenarios.jsonl \
#     --predictions local/predictions/qwen3_14b_lora_hard50.jsonl \
#     --out-metrics local/results/qwen3_14b_smoke_hard50.json \
#     --print-table
#
# Headline metrics include: format_failure_rate, memory_state_accuracy,
# evidence_f1, failure_diagnosis_accuracy, joint_revision_success,
# stale_reuse_rate.

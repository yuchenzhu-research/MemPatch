# Local MLX LoRA Fine-Tuning Preflight

This directory contains preflight configuration and data preparation utilities for open-source model replication and local training smokes on Apple Silicon (M2 Mac).

## 8 GB Unified Memory Constraints

Since this machine has **8 GB Unified Memory**, training larger models (e.g. 7B/8B parameter backbones) is **NOT** feasible locally. 

Instead, to validate the training pipeline:
1. Use a **very small quantized model** ($\le 1.5$B parameters), such as `Qwen/Qwen2.5-1.5B-Instruct` (quantized version).
2. Install the Apple Silicon optimized toolchain:
   ```bash
   pip install mlx-lm
   ```

## SFT Data Preparation

Before training, convert the reviewed development candidate dataset into MLX-compliant chat templates:
```bash
env PYTHONPATH=. python experiments/multiagent/local_training/prepare_mlx_stagec_data.py
```

## Running MLX-LM LoRA SFT Smoke

Run the MLX micro-LoRA training preflight command (do not run on actual test sets):
```bash
mlx_lm.lora \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --train \
  --data experiments/multiagent/local_training/data/ \
  --config experiments/multiagent/local_training/configs/mlx_lora_smoke.example.yaml
```

**Note**: Checkpoint outputs and adapters generated here are strictly for pipeline smoke checks and are labeled as `scientific_status = "training_pipeline_smoke_only"`.

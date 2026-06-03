# ReTrace Multi-Agent Experiments & Policies

This directory houses the evaluation entrypoints, configurations, and helper scripts for running ReTrace-Learn multi-agent method evaluations (Stage A/B/C).

## Canonical Method Layer

Following the transition to ReTrace-Bench v1.0, the core execution policies, evaluation pipelines, metrics computation, and dataset interfaces have been refactored and moved to the importable method package:

- Core Multi-Agent Evaluation Module: [`src/retracemem/evaluation/multiagent/`](file:///Users/yuchenzhu/Desktop/ReTrace/src/retracemem/evaluation/multiagent/)
  - Config contracts: [`contracts.py`](file:///Users/yuchenzhu/Desktop/ReTrace/src/retracemem/evaluation/multiagent/contracts.py)
  - Proposer pipeline: [`pipeline.py`](file:///Users/yuchenzhu/Desktop/ReTrace/src/retracemem/evaluation/multiagent/pipeline.py)
  - DirectJudge baseline: [`directjudge.py`](file:///Users/yuchenzhu/Desktop/ReTrace/src/retracemem/evaluation/multiagent/directjudge.py)
  - Validation datasets: [`data/`](file:///Users/yuchenzhu/Desktop/ReTrace/src/retracemem/evaluation/multiagent/data/) (e.g. `dev_expansion.py`, `paper1_balanced.py`)
  - Policy implementation (Stage A & C): [`stagec.py`](file:///Users/yuchenzhu/Desktop/ReTrace/src/retracemem/evaluation/multiagent/stagec.py), [`stage_c_icl.py`](file:///Users/yuchenzhu/Desktop/ReTrace/src/retracemem/evaluation/multiagent/stage_c_icl.py)

## Directory Structure

```text
experiments/multiagent/
├── run_stageab_api_eval.py       # Main evaluation entrypoint script for Stage A & Stage B
├── select_prompt_smoke_examples.py # Selects active exemplars for smoke testing
├── apply_smoke_review_decisions.py # Maps reviewed examples to smoke configs
│
├── configs/                      # Multi-agent experiment configurations
└── local_training/               # MLX LoRA training utility scripts
```

## Legacy and Archived Helpers

- Preserved baseline and comparison helper scripts (such as `fixtures.py`, `methods.py`, `metrics.py`, `episodes_dev.py`) reside strictly isolated inside [`experiments/archive/`](file:///Users/yuchenzhu/Desktop/ReTrace/experiments/archive/) and are not used in the canonical evaluation flow.
- Obsolete data generators and v2-baseline files are isolated under [`experiments/archive/obsolete_generation/`](file:///Users/yuchenzhu/Desktop/ReTrace/experiments/archive/obsolete_generation/).

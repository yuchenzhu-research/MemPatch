# ReTrace Multi-Agent Experiments & Policies

This directory contains the Stage A/B/C evaluation scripts, proposer policies, and dataset generation tools for the Paper 1 evaluation.

## Conceptual Stage Mapping

1. **Stage A (`ReTrace-API-ZeroShot` / `ReTrace-Prompt`)**
   - Main entry point: `run_stageab_api_eval.py`
   - Active proposer: `stagec_policy.py` (`ClosedAPIZeroShotProposer`, `ClosedAPIZeroShotConstrainedProposer`)
2. **Stage B (`DirectJudge-API`)**
   - Main entry point: `run_stageab_api_eval.py` (Stage B comparison baseline flow)
3. **Stage C (`ReTrace-AdaptiveProposer`)**
   - Proposer policies: `stagec_policy.py` (`ClosedAPIICLProposer`, `OpenModelPromptProposer`, `OpenModelLoRAProposer`)
   - Replay/Local Validation: `run_stagec_adapter_eval.py`

## Directory Structure

```text
experiments/multiagent/
├── run_stageab_api_eval.py       # Main evaluation script for Stage A & Stage B
├── run_model_matrix_api_eval.py  # Model matrix comparison wrapper
├── run_stagec_adapter_eval.py    # Local adapter and silver compositional evaluator
├── stagec_policy.py              # Zero-shot, ICL, and fail-closed Open policies
├── stagec_dataset.py             # Training dataset loader and verification
├── contracts.py                  # FixedCandidate and Stage C schemas
├── dev_expansion.py              # dev70 expansion dataset loader
├── episodes_fc_dev.py            # Fixed-Candidate dev episodes loader
├── select_prompt_smoke_examples.py # Active exemplar selectors for smoke validation
├── apply_smoke_review_decisions.py # Map human reviews to prompt smoke exemplars
├── testbed_spec.py               # Testbed specification metadata contracts
│
├── local_training/               # Local MLX LoRA training utility scripts
│   ├── prepare_mlx_stagec_data.py
│   └── configs/
│
└── legacy/                       # Legacy baseline and deprecated code
    ├── run_fc_comparison.py
    ├── run_offline_diagnostic.py
    └── ...
```

## Legacy Helper Modules

For backward compatibility with legacy scripts and older baseline test cases, the following helper files are preserved directly in `experiments/multiagent/`:
- `fixtures.py`: Pre-configured evaluation environment mocks.
- `methods.py`: Naive LWW, ReTrace base comparison class baselines.
- `metrics.py`: Aggregate precision/recall calculators for Paper 1 baselines.
- `episodes_dev.py`: Main dev episodes registry loader.

# ReTrace

ReTrace is a provider-agnostic shared-memory revision authorization system for multi-agent / subagent workflows.

The method core is:
```text
immutable EvidenceNode ledger
+ typed BeliefNode / ConditionNode graph
+ DependencyEdge(REQUIRES)
+ EvidenceEdge(BLOCKS / RELEASES / SUPERSEDES / REAFFIRMS / UNCERTAIN)
+ RevisionGate structural admission
+ deterministic Defeat-Path Authorization Algorithm
```

ReTrace is designed as a **pluggable authorization kernel** that controls which revisions submitted by multiple subagents are authorized to alter the shared usable memory basis.

---

## Academic Stage Definitions

1. **Stage A (`ReTrace-API-ZeroShot` / `ReTrace-Prompt`)**
   Proposes typed revision action proposals (e.g. `SUPERSEDES`, `BLOCKS`, etc.) using zero-shot prompting over candidate belief contexts, validated by `RevisionGate` and authorized deterministically via Defeat-Path Authorization (DPA).
2. **Stage B (`DirectJudge-API`)**
   Baseline pathway that directly predicts final usability verdicts (`USABLE`, `NOT_USABLE`, `UNCERTAIN`) for all candidate beliefs without edge decomposition.
3. **Stage C (`ReTrace-AdaptiveProposer`)**
   Explicit typed revision proposal policy learning (e.g., API-ZeroShot, API-ICL, LoRA-SFT).

> [!NOTE]
> Latent-memory representation, long-horizon consolidation, and RL over hidden memory states belong strictly to Paper 2, not ReTrace Paper 1.

---

## Pluggable DPA Precedence

For any candidate belief $b$:
$$A_t(b) = \text{DPA}(b, S_t) \in \{\text{AUTHORIZED}, \text{BLOCKED}, \text{SUPERSEDED}, \text{UNRESOLVED}\}$$
with canonical precedence:
$$\text{SUPERSEDES} > \text{PREREQUISITE\_BLOCK} > \text{UNRESOLVED\_UNCERTAIN} > \text{AUTHORIZED}$$

---

## Execution & Commands Guide

### Offline Verification & Compilation
Before running any scripts, verify the codebase compiles and passes tests:
```bash
# Compilation check
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests experiments

# Run offline unit/integration tests
.venv/bin/python -m pytest
```

### Stage A/B dev70 Evaluation (Dry-run & Live)
Run evaluation on the dev70 expansion dataset:
```bash
# Safe Dry-run/Mock Mode (runs local gold simulation)
.venv/bin/python experiments/multiagent/run_stageab_api_eval.py --mock --dataset dev_expansion

# Live API Evaluation (requires SILICONFLOW_API_KEY)
.venv/bin/python experiments/multiagent/run_stageab_api_eval.py --live --provider siliconflow --model deepseek-ai/DeepSeek-V3
```

---

## Development Utilities

The following tools and evaluations are auxiliary utilities for model comparisons, adapter scoring, and verification experiments. They are not the primary experimental boundaries.

### 1. Model Matrix Evaluator (Dry-run & Live)
Compare multiple models and methods (DirectJudge, StageA-Freeform, StageA-Constrained, StageC-ICL) in a single run:
```bash
# Dry-run Mock Run
.venv/bin/python experiments/multiagent/run_model_matrix_api_eval.py --dry-run

# Live API Run (SiliconFlow)
.venv/bin/python experiments/multiagent/run_model_matrix_api_eval.py
```

### 2. Stage C Adapter Evaluator
Evaluate base models vs LoRA adapters across structural subsets:
```bash
# Print CLI options and subsets help
.venv/bin/python experiments/multiagent/run_stagec_adapter_eval.py --help
```

### 3. STALE-style Synthetic Validation
Run synthetic temporal sequence validation comparing Append-only profile, Oracle, and ReTrace methods:
```bash
.venv/bin/python experiments/stale_style_retrace_validation.py
```

---

## Governance Policy

All generated summaries and run logs reside in `outputs/` or `outputs/runs/` and are strictly excluded from git.
Checkpoints, adapter weights (`checkpoints/`, `adapters/`), and API caches are excluded to guarantee reproducibility and prevent key leakage.

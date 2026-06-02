# ReTrace-Learn — Paper Skeleton

## Title Placeholder

> ReTrace-Learn: Trainable Typed Revision Proposal for Verifiable Shared Memory

## Abstract Placeholder

One paragraph: multi-agent systems need shared long-term memory, but updates
from different subagents can stale, conflict, or violate scope. ReTrace-Learn
turns revision authorization into a verifiable learning problem: models extract
typed graph state and propose typed revision actions, while a deterministic
Authorization Court admits valid effects and computes final memory eligibility.

## 1 Introduction

- Shared-memory revision is distinct from retrieval, storage, and debate.
- The LLM proposes; deterministic authorization decides.
- Contributions: graph extraction, typed revision proposal, DPA-in-the-loop
  feedback, fair Prompt-Proposer / DirectJudge / ReTrace-Learn comparison.

## 2 Problem Formulation

- Multi-subagent submissions, candidate memory view, evidence-bearing updates.
- Distinguish typed actions from final DPA statuses.
- Source: `docs/retrace_learn_positioning.md`, `docs/retrace_learn_pipeline.md`.

## 3 Method

- Graph Extractor.
- Typed Revision Proposer.
- Authorization Court / ReTrace-Engine via `authorize(...)`.
- Source: `docs/architecture.md`, `docs/retrace_learn_pipeline.md`.

## 4 Training And Feedback

- SFT / RSFT / DPO over explicit typed actions.
- Deterministic DPA traces as training/evaluation feedback.
- Human-approved example boundary for live smoke/training export.

## 5 Experiments

- E0: Oracle/replay kernel validation.
- E1: Fixed-candidate revision evaluation.
- E2: Stage C training and model-driven proposal evaluation.
- E3: Closed-loop multi-agent workflow.
- E4: STALE/CUPMem external validation through isolated adapters.

## 6 Results And Analysis

- Stage A / Stage B / Stage C comparison over identical contexts.
- Typed-action metrics, grounding errors, and final-status accuracy.
- Ablations by action type and multi-action composition.

## 7 Limitations

- No latent memory or hidden-state consolidation.
- External validation is mapped through isolated interfaces.
- Live API paths are provider-agnostic but separate from deterministic commit.

## 8 Reproducibility

- `scripts/evaluate.py` for Stage A/B/C.
- `outputs/` run manifests for provenance.
- Compile and full offline test commands from `README.md`.

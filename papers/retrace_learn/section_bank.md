# ReTrace-Learn — Section Bank

## Core Claim

ReTrace-Learn turns shared-memory revision authorization into a verifiable
learning problem. Models learn to propose explicit, typed revision actions over
method-visible candidate structure, and the deterministic Authorization Court
decides which proposed effects are admitted and which beliefs remain eligible
for current answers.

## Method Summary

The pipeline is:

```text
Raw multi-subagent content
    -> Graph Extractor
    -> candidate evidence / belief / condition graph
    -> Typed Revision Proposer
    -> typed revision actions
    -> Authorization Court / ReTrace-Engine
    -> final statuses + audit trace
```

Source: `docs/retrace_learn_pipeline.md`, `docs/architecture.md`.

## Evaluation Summary

Stage A (`ReTrace-Prompt`) and Stage C (`ReTrace-Learn`) are typed-action
proposer families. Stage B (`DirectJudge-API`) is a direct final-status baseline
that bypasses the engine. Fair comparisons use identical candidate contexts and
separate proposer quality, grounding, parser/gate failures, and final DPA status
accuracy.

Source: `docs/experiment_protocol.md`, `docs/evaluation.md`.

## Boundary Summary

The method paper excludes latent memory, learned forgetting, hidden-state
consolidation, and RL over hidden memory states. STALE/CUPMem are external
validation pathways, not the method definition.

Source: `docs/retrace_learn_positioning.md`, `AGENTS.md`.

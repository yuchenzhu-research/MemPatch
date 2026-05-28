# ReTrace

ReTrace is the working codebase for ICLR 2027 Paper 1:

> Evidence-preserving reversible authorization for evolving agent memory.

ReTrace is not latent memory learning, RL memory-action training, or a generic
memory framework clone. It preserves original evidence and changes whether a
belief may govern current answers through typed, auditable defeat paths.

The method core is:

```text
immutable EvidenceNode ledger
→ BeliefNode / ConditionNode graph
→ DependencyEdge(REQUIRES)
→ EvidenceEdge(BLOCKS / RELEASES / SUPERSEDES / REAFFIRMS / UNCERTAIN)
→ RevisionGate structural admission
→ deterministic Defeat-Path Authorization Algorithm
```

## Current Status

The active branch for the current feasibility packet is
`experiment/retrace-ab-feasibility`, created from
`integration/retrace-v1-complete` @
`5e8d6e2d1a494d572d6d0fa929595bb198154390`.

Validated so far:

- typed DPA execution spine;
- offline Stage A/B contracts, prompts, sibling DirectJudge path, and mock/replay tests;
- fairness and deterministic-grounding hardening;
- offline controlled attribution harness;
- auditability and comparison-protocol lock;
- AB-1B replay-only internal development evaluation;
- Stage A v1 effect-triggered prompt semantics and the hard_v1 internal
  Ambiguity-and-Scope challenge split for exploratory feasibility.

Implemented scaffolding that must not be overclaimed:

- provider/cache/manifest infrastructure;
- secondary end-to-end development runner;
- STALE and Memora adapter smoke/dry-run entrypoints;
- Stage C deferral report (`docs/stage_c_report.md`).

No current repository result establishes:

- Stage A superiority over Stage B;
- verified live provider performance;
- official STALE or Memora scores;
- paper-facing retrieval validity;
- Stage C training labels.

Stage A v1 treats existing beliefs as already evidence-supported. New evidence
changes authorization only by producing a direct local typed effect; irrelevant
or silent evidence should produce `{"edges": []}` and preserve authorization.
`UNCERTAIN` is reserved for directly relevant but unresolved updates.

## Canonical Docs

- `AGENTS.md`: first-read instructions for coding models.
- `docs/method_spec_dpa.md`: technical authority for runtime semantics.
- `docs/stage_ab_protocol.md`: active Stage A/B protocol authority.
- `docs/paper1_blueprint_zh.md`: canonical Chinese scientific blueprint.
- `docs/repository_execution_contract.md`: reproducibility and handoff contract.
- `docs/coding_contract.md`: package-boundary and editing rules.
- `docs/implementation_status.md`: concise live repository status.
- `docs/stage_c_report.md`: Stage C go/no-go recommendation report.
- `docs/upstream_integration.md`: upstream roles and clean-room integration.

## How to Run

### Compile and Run Tests
```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests scripts
.venv/bin/python -m pytest
```

### Run End-to-End Pipeline Dev Test
```bash
.venv/bin/python scripts/run_end_to_end_dev.py
```

### Run Ambiguity-and-Scope Replay Diagnostics
```bash
.venv/bin/python scripts/run_ambiguity_scope_ab_dev.py --mode replay
.venv/bin/python scripts/run_ambiguity_scope_ab_dev.py --mode replay --pilot-only
.venv/bin/python scripts/run_ambiguity_scope_ab_dev.py \
  --mode replay \
  --dataset data/internal_dev/ambiguity_scope_hard_v1.json \
  --case-ids ash_dense_01,ash_temporal_01,ash_premise_01,ash_release_01,ash_uncertain_01,ash_cross_01 \
  --skip-balance-check \
  --run-type replay_correctness \
  --stage-a-prompt-version evidence_edge_prediction_v1
```

The original simple pilot is regression-only. The hard_v1 split is internal
exploratory development material, not an official benchmark or paper result.

### Run STALE Adapter Smoke/Dry-Run
```bash
.venv/bin/python scripts/run_stale_official_eval.py --limit 1
```

This is not an official STALE benchmark result unless a later frozen evaluation
task explicitly authorizes and validates it.

### Run Memora Adapter Smoke/Dry-Run
```bash
.venv/bin/python scripts/run_memora_official_eval.py --limit 1
```

This is not an official Memora benchmark result unless a later frozen evaluation
task explicitly authorizes and validates it.

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

The active branch is `method/retrace-llm-directjudge`.

Completed through repaired Stage AB-1B:

- typed DPA execution spine;
- offline Stage A/B contracts, prompts, sibling DirectJudge path, and
  mock/replay tests;
- fairness and deterministic-grounding hardening;
- offline controlled attribution harness;
- auditability and comparison-protocol lock;
- replay-only internal evaluation with repaired cost/metric/provenance
  semantics.

No live-provider result, official STALE result, official Memora result, or
Stage C result is claimed yet.

## Canonical Docs

- `AGENTS.md`: first-read instructions for coding models.
- `docs/method_spec_dpa.md`: technical authority for runtime semantics.
- `docs/stage_ab_protocol.md`: active Stage A/B protocol authority.
- `docs/paper1_blueprint_zh.md`: canonical Chinese scientific blueprint.
- `docs/repository_execution_contract.md`: reproducibility and handoff
  contract.
- `docs/coding_contract.md`: package-boundary and editing rules.
- `docs/implementation_status.md`: concise live repository status.
- `docs/upstream_integration.md`: upstream roles and clean-room integration.

## Verification

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests scripts
.venv/bin/python -m pytest
```

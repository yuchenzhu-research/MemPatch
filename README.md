# ReTrace

ReTrace is a research codebase for evidence-preserving reversible authorization in evolving agent memory.

The method core is:

```text
immutable EvidenceNode ledger
+ typed BeliefNode / ConditionNode graph
+ DependencyEdge(REQUIRES)
+ EvidenceEdge(BLOCKS / RELEASES / SUPERSEDES / REAFFIRMS / UNCERTAIN)
+ RevisionGate structural admission
+ deterministic Defeat-Path Authorization Algorithm
```

ReTrace is not generic RAG, a Mem0/Graphiti clone, RL memory-action learning, latent memory training, or an unconstrained LLM memory judge.

## Implemented Core

- Deterministic DPA over admitted typed edges.
- Stage A `ReTrace-LLM`: local typed-edge proposal plus RevisionGate plus DPA.
- Stage B `DirectJudge-LLM`: direct shared-view adjudication baseline.
- Per-belief Stage A controlled reference path.
- Batched Stage A scalable development path.
- Bounded batched backend ingestion.
- Official frozen STALE adapter and offline non-leaking Stage A/B wiring demo.
- Memora oracle-conditioned authorization diagnostic, retained as a rejected
  internal pilot artifact.

## Current Status

Dynamic branch, HEAD, smoke, and validation status live only in
`docs/implementation_status.md`.

## Canonical Docs

- `AGENTS.md`: stable coding-agent instructions.
- `docs/method_spec_dpa.md`: method semantics.
- `docs/stage_ab_protocol.md`: Stage A/B protocol.
- `docs/implementation_status.md`: current dynamic status and entrypoints.
- `docs/upstream_integration.md`: clean-room upstream boundaries.

## Offline Validation

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests scripts
.venv/bin/python -m pytest
```


## Current Offline STALE Wiring Demo

```bash
.venv/bin/python scripts/run_stale_official_frozen_eval.py --limit-t1 2 --limit-t2 2
```

This consumes the public official frozen STALE dataset
`STALEproj/STALE::T1_T2_400_FULL.json` from the gitignored local path
`data_external/stale_official_frozen/`. It validates non-leaking method wiring
and official answer export schema only. It is not an official STALE model
result and does not run the official judge.

## Memora Negative Pilot

The Memora oracle-conditioned diagnostic is retained only as an internal
rejected-pilot artifact demonstrating adapter/objective mismatch. It is not
official end-to-end Memora evaluation, not FAMA, and not a paper result.

## Non-Claims

The repository currently does not establish:

- Stage A superiority over Stage B;
- official STALE or Memora scores;
- paper-facing retrieval validity;
- Stage C training labels;
- benchmark-general live-provider performance.

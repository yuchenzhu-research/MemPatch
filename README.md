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
- Memora oracle-conditioned authorization diagnostic.

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


## Current Diagnostic Replay

```bash
.venv/bin/python scripts/run_memora_development_eval.py --mode replay --period weekly --persona academic_researcher --limit-questions 2 --stage-a-execution batched
```


This is a Memora Oracle-Conditioned Authorization Diagnostic. Candidate beliefs
originate from Memora evaluation annotations (`memory_evidence` /
`forgetting_evidence`). It is not official end-to-end Memora evaluation and is
not a paper result.

## Non-Claims

The repository currently does not establish:

- Stage A superiority over Stage B;
- official STALE or Memora scores;
- paper-facing retrieval validity;
- Stage C training labels;
- benchmark-general live-provider performance.

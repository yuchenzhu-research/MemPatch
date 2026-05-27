# Agent Handoff

Read this before making changes.

## One-Sentence Project Alignment

ReTrace preserves original evidence and revises only the currently authorized
belief view through traceable local defeat paths.

## Do Not Reinterpret The Task

Do not turn this codebase into:

- generic RAG;
- a Mem0 clone;
- a Graphiti clone;
- CUPMem fixed-slot state tracking;
- RL memory action learning;
- latent memory consolidation;
- a new benchmark generator.

## Current First-Version Goal

Finish the smallest research-grade loop:

```text
BoundaryAudit JSONL
→ heuristic relation verifier
→ ReTrace pipeline
→ TMS authorization
→ authorized basis
→ unified JSONL output
→ smoke metrics
```

Then add STALE and Memora smoke runners using existing adapters.

## Existing Stable Contracts

Do not break these without updating all tests and docs:

- `retracemem.schemas.EpisodicEvidence`
- `retracemem.schemas.Belief`
- `retracemem.schemas.RelationPrediction`
- `retracemem.schemas.AuthorizationDecision`
- `retracemem.schemas.EvaluationRecord`
- backend interface in `retracemem.backends.base`
- JSONL helpers in `retracemem.evaluation.jsonl`

## Implementation Rules

- Keep code minimal.
- Use standard library only unless explicitly instructed.
- Do not call LLM APIs in core logic.
- Keep benchmark-specific logic inside adapters or runners.
- Do not edit `reference/`.
- Do not commit outputs or caches.
- Add tests for every new behavior.
- Run compileall before committing.

## Best Next Task

If no other instruction is given, implement these in order:

1. `data/boundary_audit/minimal.jsonl`
2. `retracemem/verifier/heuristic_verifier.py`
3. `retracemem/pipeline.py`
4. `scripts/run_boundary_audit.py`
5. tests for heuristic verifier and pipeline
6. smoke improvements for `scripts/run_stale.py`
7. smoke improvements for `scripts/run_memora.py`

## Expected Commit Style

Use short English commit messages:

- `Add BoundaryAudit diagnostic cases`
- `Add heuristic relation verifier`
- `Add ReTrace pipeline`
- `Add BoundaryAudit runner`
- `Add benchmark smoke runners`


# ReTrace Agent Instructions

This file is the first file every coding model should read before editing this
repository.

## Mandatory Context Stack

Read these files before making code changes:

1. `docs/model_context_index.md`
2. `docs/source_materials/iclr_2027_paper_1_final_blueprint_re_trace.md`
3. `docs/source_materials/re_trace_companion_codebase_integration_and_model_handoff.md`
4. `docs/project_logic.md`
5. `docs/coding_contract.md`
6. `docs/implementation_status.md`
7. `docs/reference_integration_map.md`

The two files under `docs/source_materials/` are raw source materials copied from
the project planning stage. Do not edit them. If they conflict with current
repository style, follow `docs/model_context_index.md` and
`docs/coding_contract.md`.

## One-Sentence Alignment

ReTrace preserves original evidence and revises only the currently authorized
belief view through traceable local defeat paths.

## Do Not Drift

Do not turn this codebase into:

- generic RAG;
- a Mem0 clone;
- a Graphiti clone;
- CUPMem fixed-slot state tracking;
- RL memory action learning;
- latent memory consolidation;
- a new benchmark generator.

## Current First-Version Shape

The current runnable loop is:

```text
BoundaryAudit JSONL
→ HeuristicRelationVerifier
→ ReTracePipeline
→ RevisionGate
→ AuthorizationEngine / BasisBuilder
→ EvaluationRecord JSONL
```

Benchmark smoke runners exist for STALE and Memora. They are smoke checks, not
paper-scale evaluation yet.

## Coding Rules

- Standard library first.
- Keep core logic API-free and deterministic.
- Keep benchmark-specific logic in adapters or runners.
- Do not edit `reference/`.
- Do not commit `reference/`, `outputs/`, caches, local environments, or API
  keys.
- Preserve the dataclass contracts in `retracemem/schemas.py`.
- All methods should emit `EvaluationRecord` or JSON-compatible records.
- Add or update tests for every new behavior.

## Verification

No-dependency verification:

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile python3 -m compileall -q retracemem tests scripts
```

If pytest is installed:

```bash
python3 -m pytest -q
```

## Commit Style

Use short English commits with production-level scope:

- `Add ...`
- `Implement ...`
- `Document ...`
- `Wire ...`
- `Fix ...`

Do not bundle unrelated method, runner, and documentation changes into one
commit unless the change is purely mechanical.


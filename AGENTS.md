# ReTrace Agent Instructions

This file is the first file every coding model should read before editing this
repository.

## Mandatory Context Stack

Read these files before making code changes:

1. `docs/refactor_plan_defeat_path.md` (Governs refactor implementation and DPA core specifications)
2. `docs/model_context_index.md`
3. `docs/project_logic.md`
4. `docs/coding_contract.md`
5. `docs/implementation_status.md`
6. `docs/reference_integration_map.md`
7. `docs/source_materials/iclr_2027_paper_1_final_blueprint_re_trace.md`
8. `docs/source_materials/re_trace_companion_codebase_integration_and_model_handoff.md`

Note: The two files under `docs/source_materials/` are raw historical planning documents copied from the design stage. They are preserved for original research scope and motivation, but their superseded implementation vocabulary (flat relation labels, legacy pipeline) is outranked by `docs/refactor_plan_defeat_path.md` and current verifier contract specifications.

## One-Sentence Alignment

ReTrace preserves immutable evidence and changes a belief's eligibility for current answers only through verified, temporally valid typed defeat paths computed by deterministic DPA.

## Do Not Drift

Do not turn this codebase into:

- generic RAG;
- a Mem0 clone;
- a Graphiti clone;
- CUPMem fixed-slot state tracking;
- RL memory action learning;
- latent memory consolidation;
- a new benchmark generator;
- Do not present heuristic keyword fixtures as the publishable ReTrace method.
- Do not let legacy RelationPrediction / CONDITION / REQUIRED_BY semantics govern new runtime code.
- Do not claim that DPA eliminates semantic model judgment; it constrains local edge prediction and makes final authorization deterministic.

## Current Method Status

Typed graph schemas and DPA logic exist.
Canonical EvidenceNode ledger migration and typed end-to-end integration are not yet complete.

Canonical typed DPA core exists:
- EvidenceNode / BeliefNode / ConditionNode
- DependencyEdge(REQUIRES)
- EvidenceEdge(BLOCKS / RELEASES / SUPERSEDES / REAFFIRMS / UNCERTAIN)
- DefeatPathAuthorizationAlgorithm
- typed verifier contracts (`RequirementProposal`)

Not yet integrated (Wave 2+):
- typed backend ingestion
- query-conditioned authorized basis
- generic ReTrace-LLM semantic edge predictor
- DirectJudge-LLM attribution baseline
- official frozen benchmark evaluation

Development-only:
- heuristic requirement inducer and evidence-edge verifier
- hand-written gate and contract unit tests

## Coding Rules

- Standard library first.
- Keep core logic API-free and deterministic.
- Keep benchmark-specific logic in adapters or runners.
- Do not edit `reference/`.
- Do not commit `reference/`, `outputs/`, caches, local environments, or API
  keys.
- Preserve the dataclass contracts in `retracemem/schemas.py`.
- All new method paths should emit JSON-compatible typed traces or score records with provenance. Legacy EvaluationRecord is transitional only and must not govern new runtime design.
- Add or update tests for every new behavior.

## Verification

To run compilation check:

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests scripts
```

To run test suites:

```bash
.venv/bin/python -m pytest
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



# ReTrace Agent Instructions

This file is the first file every coding model should read before editing this
repository.

## Mandatory Context Stack

Read these files before making code changes:

1. `docs/refactor_plan_defeat_path.md` (Governs refactor implementation and DPA core specifications)
2. `docs/stage_ab_retrace_llm_directjudge_plan.md` (Governs Stage A/B method development)
3. `docs/model_context_index.md`
4. `docs/project_logic.md`
5. `docs/coding_contract.md`
6. `docs/implementation_status.md`
7. `docs/reference_integration_map.md`
8. `docs/source_materials/iclr_2027_paper_1_final_blueprint_re_trace.md`
9. `docs/source_materials/re_trace_companion_codebase_integration_and_model_handoff.md`

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

Wave 2 typed execution spine is complete and full-test closed (129 passed).

Implemented canonical runtime:
- EvidenceNode ledger (append-only, typed);
- BeliefNode / ConditionNode typed graph;
- DependencyEdge(REQUIRES);
- EvidenceEdge(BLOCKS / RELEASES / SUPERSEDES / REAFFIRMS / UNCERTAIN);
- DefeatPathAuthorizationAlgorithm;
- typed extraction and impact/query retrieval contracts;
- query-conditioned authorized basis;
- offline audit-preserving backend and pipeline;
- explicit canonical constructors with development-only fixture factories.

Development fixtures (heuristic/manual extractors, inducers, verifiers, retrievers) are test/smoke only and forbidden for paper main-result runners.

Next implementation stage must build Stage A and Stage B together:
- Stage A ReTrace-LLM: main typed-edge prediction plus DPA method (generic semantic extraction, requirement induction, evidence-edge prediction, deterministic DPA).
- Stage B DirectJudge-LLM: matched same-model direct-adjudication attribution baseline, implemented as a sibling method path (not an EvidenceEdgeVerifier).
- Stage C ReTrace-Local: later learned local typed-edge verifier using the same DPA; begins only after Stage A/B establish that the structured DPA formulation has value.

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



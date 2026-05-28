# Implementation Status

Last updated: 2026-05-28

This document tracks the first research-code version. It intentionally has only
two sections:

- Total plan: the agreed implementation surface for the first complete version.
- Completed: what is already implemented in this repository.

Anything in the total plan that is not listed as completed is remaining work by
definition.

## Total Plan

### Research Alignment

- Lock ReTrace as evidence-preserving reversible belief revision.
- Keep external reference repositories isolated under `reference/`.
- Keep Paper 1 focused on STALE, Memora, and a small BoundaryAudit diagnostic
  set.
- Avoid latent memory, RL/GRPO, destructive consolidation, and fixed-slot
  ontology drift.

### Core Data Contracts

- Stable dataclass schemas for EvidenceNode, BeliefNode, ConditionNode, DependencyEdge, EvidenceEdge, and evaluation records.
- Unified JSONL output for all methods.
- Cost/call tracking helpers.

### Memory Core

- Append-only `EpisodeLedger`.
- Open-text `BeliefStore`.
- Conservative `RevisionGate`.
- `DefeatPathAuthorizationAlgorithm` for deciding whether a belief can govern current
  answers.
- `query-conditioned basis` construction at query-time.

### Verifier Layer

- Deterministic `RequirementInducer` and `EvidenceEdgeVerifier` for local smoke runs.
- `RequirementProposal` and typed verifier contracts.
- Evidence edge verifiers including `ReTrace-LLM`, `DirectJudge-LLM`, and `ReTrace-Local`.
- Edge types:
  - `DependencyEdge(REQUIRES)`
  - `EvidenceEdge(BLOCKS, RELEASES, SUPERSEDES, REAFFIRMS, UNCERTAIN)`
- Fail-closed behavior for ambiguous or empty inputs.

### ReTrace Pipeline

- `ReTracePipeline` that combines:
  - evidence ledger;
  - belief store;
  - verifier;
  - revision gate;
  - query-conditioned basis;
  - deterministic evaluation record output.

### BoundaryAudit Diagnostic Loop

- 20 local JSONL diagnostic cases.
- Balanced case buckets covering `SUPERSEDES`, `BLOCKS`, `RELEASES`, `REAFFIRMS`, `UNCERTAIN`, and dependency prerequisites.
- Runner for:
  - `retrieval_baseline`
  - `retrace_heuristic`
- Summary metrics:
  - `cases_total`
  - `relation_correct`
  - `authorization_correct`
  - `protected_beliefs_preserved`
  - `unsupported_revision_count`

### Benchmark Smoke Runners

- STALE smoke runner:
  - discover `*_MAIN.json`;
  - load a limited number of samples;
  - run retrieval baseline;
  - write unified JSONL.
- Memora smoke runner:
  - discover persona roots;
  - load chronological sessions and questions;
  - run retrieval baseline;
  - write unified JSONL.
- Empty/missing benchmark data should exit cleanly with an empty JSONL file.

### Tests And Verification

- No-dependency compile check:

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile python3 -m compileall -q retracemem tests scripts
```

- Unit tests for:
  - adapters;
  - memory core;
  - TMS authorization;
  - retrieval baseline;
  - evaluation helpers;
  - heuristic verifier;
  - ReTrace pipeline;
  - runners.

## Completed

### Research Alignment

- Added `AGENTS.md`.
- Added `docs/model_context_index.md`.
- Added `docs/project_logic.md`.
- Added `docs/coding_contract.md`.
- Added `docs/agent_handoff.md`.
- Added `docs/today_execution_plan.md`.
- Added `docs/reference_integration_map.md`.
- Added `docs/code_direction.md`.
- Added `docs/refactor_plan_defeat_path.md`.
- Added raw source materials under `docs/source_materials/`.

### Wave 0 (Refactor Plan & Primitives)
- Defined locked refactor plan with amendments A1-A10.
- Implemented canonical typed graph schemas: `EvidenceNode`, `BeliefNode`, `ConditionNode`, `DependencyEdge`, `EvidenceEdge`, `DefeatPath`, `AuthorizationTrace` in `schemas.py`.
- Added round-trip tests in `tests/test_schema_roundtrip.py`.

### Wave 1A (Typed Graph Store & DPA Core)
- Updated `BeliefStore` index collections (in `memory/belief_store.py`) to manage typed nodes and edges.
- Implemented `RevisionGate` (in `tms/gate.py`) and `DefeatPathAuthorizationAlgorithm` (in `tms/authorization.py`) mapping DPA rules.
- Created `tests/gate_unit/` verification suite for DPA logic.

### Wave 1B (Typed Verifier Contracts)
- Introduced `RequirementProposal` dataclass and updated contracts protocol signature in `contracts.py`.
- Implemented `HeuristicRequirementInducer` and `HeuristicEvidenceEdgeVerifier` as development-only deterministic fixtures.
- Updated `tests/verifier_contract/` verification suite.

### Hotfix and Environment Alignment (Current Status)
- Aligned `EvidenceEdgeVerifier.verify_edges` signature.
- Rewrote Heuristic Inducer condition texts to use atomic prerequisite proposition semantics.
- Recreated project-local `.venv` using Python 3.10.20 and editable install layout.
- Ignored egg-info and pytest cache metadata.
- Isolated test cache artifacts to pytest `tmp_path`.

### Legacy Modules (Remaining Migration Tasks)
- Typed belief/condition/evidence-edge graph structures and DPA logic are implemented.
- EpisodeLedger and TemporalValidity still use legacy EpisodicEvidence entries and must migrate to EvidenceNode in Wave 2.
- Backend, pipeline, extraction, retrieval, and query-conditioned basis also remain Wave 2 integration work.
- Obsolete HeuristicRelationVerifier pipeline results are archived as prototype-only milestones.

### Tests And Verification

- Passed test suites (with Python 3.10.20 and pytest 9.0.3):
  * `tests/test_schema_roundtrip.py`
  * `tests/gate_unit/`
  * `tests/verifier_contract/`
- Verified local compile check:
  ```bash
  env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests scripts
  ```
- Command to run verification tests:
  ```bash
  .venv/bin/python -m pytest -q tests/test_schema_roundtrip.py tests/gate_unit tests/verifier_contract
  ```

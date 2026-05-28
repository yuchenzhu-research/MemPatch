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

### Wave 2 (Pipeline Closure & Legacy Test Migration)
- `ReTraceBackend`: removed unused `model_id` / `provider` parameters.
- `ReTracePipeline`: requires explicit `backend` or all five typed components; no silent fixture default. Added `for_development_fixture()` classmethod.
- `ReTracePipeline.answer()`: uses query-conditioned `search()` excluded traces instead of scanning all beliefs.
- Migrated `test_memory_core.py` to typed `BeliefNode` / `EvidenceNode` schemas.
- Migrated `test_rollback_diagnostics.py` to typed graph construction (no flat fixture dependency).
- Retired `test_tms_authorization.py` (covered by `tests/gate_unit/`).
- Retired `test_pipeline.py` (covered by `tests/backend_contract/`; JSONL test migrated).
- Added regression tests: `test_pipeline_requires_explicit_backend_or_all_components`, `test_pipeline_answer_blocked_beliefs_are_query_conditioned`, `test_pipeline_answer_record_is_jsonl_compatible`.
- Rewrote `scripts/run_retrace_internal_dev.py` to use typed fixtures only (no API clients, no legacy types).
- Updated `tests/test_runners.py` to verify typed fixture banner.

### Legacy Modules (Remaining Migration Tasks)
- Typed belief/condition/evidence-edge graph structures, DPA logic, typed pipeline, and dev runner are implemented.
- Obsolete HeuristicRelationVerifier pipeline results are archived as prototype-only milestones.

### Next Stage

- Wave 2 is merged into `main` (commit `dcf121b`).
- Stage A/B planning branch `method/retrace-llm-directjudge` has begun.
- See `docs/stage_ab_retrace_llm_directjudge_plan.md` for full details.
- **Stage AB-0 contracts and offline mock/replay infrastructure are implemented.**
  - Shared `SharedCandidateView` controlled-comparison contracts in `methods/contracts.py`.
  - `PromptTypedBeliefExtractor`, `PromptRequirementInducer`, `PromptEvidenceEdgeVerifier` in `verifier/`.
  - `DirectJudgeLLM` in `methods/directjudge.py` as a sibling method path (not an EvidenceEdgeVerifier).
  - Versioned prompt templates in `prompts/retrace_llm/` and `prompts/directjudge/`.
  - All tested offline with `MockLLMProvider` and replay cache; no live API calls or benchmark evaluations have occurred.
- **Stage AB-0.5 fairness and replay-determinism hardening is complete.**
  - DirectJudge consumes the full `SharedCandidateView` (evidence, beliefs, replacements, conditions).
  - DirectJudge enforces exactly one verdict per candidate belief; omissions and duplicates are parser failures.
  - `PromptRequirementInducer` derives scope from `belief.metadata["scope_id"]` with no fallback.
  - `SUPERSEDES` replacements must be grounded in the current `EvidenceNode`.
  - All graph ids (belief_id, condition_id, edge_id) are computed deterministically from grounded inputs.
  - `SharedCandidateView.__post_init__` validates uniqueness and key consistency.
  - No live API call, provider dependency, official evaluation, or DPA/backend change occurred.
- **Stage AB-1A offline controlled attribution harness is complete.**
  - `SharedCandidateView` extended with `new_evidence`, `dependency_edges_by_belief` (immutable tuple-of-tuples), and deterministic `view_fingerprint`.
  - `ControlledReTraceLLM` runner: consumes fixed view → PromptEvidenceEdgeVerifier → isolated typed graph → RevisionGate → DPA → ControlledMethodResult.
  - No extraction, induction, or retrieval in the primary controlled comparison.
  - DirectJudge and ControlledReTrace share the same `view_fingerprint` in provenance.
  - Both methods report per-instance cost (delta accounting), not cumulative totals.
  - Primary status mapping: AUTHORIZED→USABLE, BLOCKED/SUPERSEDED→NOT_USABLE, UNRESOLVED→UNCERTAIN.
  - No live API call, provider SDK, official evaluation, or Stage C work occurred.
- **Stage AB-1A.5 auditability and comparison protocol lock is complete.**
  - `SharedCandidateView` now has mandatory `new_evidence` (no longer optional); `view_fingerprint` is `init=False` (derived, not caller-settable).
  - Canonical versioned JSON + SHA-256 fingerprint covering all first-class fields of EvidenceNode, BeliefNode, ConditionNode, DependencyEdge. Metadata excluded (non-semantic policy documented).
  - New invariants: duplicate evidence_ids rejected, new_evidence payload identity check, candidate/replacement overlap rejected, repeated condition/dep keys rejected, conflicting condition payloads rejected.
  - `EdgePredictionBatch` contract: preserves `model_call_trace_id` even for zero-edge verifier invocations.
  - `PromptEvidenceEdgeVerifier.verify_edges_with_trace()` returns traced batch; backward-compatible `verify_edges()` delegates.
  - `ControlledReTraceLLM`: uses traced verifier, fails loudly on rejected fixed DependencyEdge anchors, records admitted anchors and edge proposal provenance (admitted/rejected + gate reason). Records model_revision_or_api_version.
  - `DirectJudgeLLM`: upgraded to prompt v1 with explicit new_evidence identity/timestamps/source. Records model_revision_or_api_version in provenance.
  - Honest protocol claims: Stage A calls N times (one per candidate belief for edge prediction); Stage B calls once (direct adjudication). Prompts differ by design.
  - No live API calls, no real provider adapters, no Stage C, no backend/pipeline/DPA core changes.
- Stage AB-1B will construct internal development cases and add one real provider adapter after approval.

- **Stage A: ReTrace-LLM** — main generic typed-edge prediction plus DPA method. Replaces all development-only heuristic/manual fixtures for paper main-result runs. Components: generic typed belief extraction, generic requirement/condition induction, generic evidence-edge prediction, existing deterministic DPA and authorized-basis pipeline.
- **Stage B: DirectJudge-LLM** — matched same-model final-adjudication attribution baseline, implemented alongside Stage A as a sibling method path. DirectJudge-LLM is **not** an `EvidenceEdgeVerifier`; it directly decides memory usability without DPA, using the same model family as Stage A. Call cardinalities differ by design: Stage A makes N calls (one per candidate belief for edge prediction); Stage B makes one call (direct adjudication over all candidates).
- **Stage C: ReTrace-Local** — later learned local typed-edge verifier variant (SFT / LoRA) using the same DPA core. Deferred until Stage A/B validation establishes that the structured DPA formulation has value.

### Tests And Verification

- Full green suite: 224 tests passed (Python 3.10.20, pytest 9.0.3).
- Passed test suites:
  * `tests/test_schema_roundtrip.py`
  * `tests/gate_unit/`
  * `tests/verifier_contract/`
  * `tests/backend_contract/`
  * `tests/method_contract/`
  * `tests/test_memory_core.py`
  * `tests/test_rollback_diagnostics.py`
  * `tests/test_runners.py`
- Verified local compile check:
  ```bash
  env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests scripts
  ```
- Command to run full verification:
  ```bash
  .venv/bin/python -m pytest
  ```

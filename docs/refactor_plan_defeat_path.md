# Refactor Plan: Evidence-Grounded Defeat-Path Authorization

Status: locked. Amendments A1-A10 confirmed 2026-05-28.
Branch: `refactor/defeat-path-core` off `main` HEAD `67cc630`.
Authority: governs all code changes during the refactor; ranks above
`docs/implementation_status.md` for any conflict until merged back.

## 0. Non-goals and invariants

- Do not drift toward RAG, Mem0 clones, fixed-slot tracking, or RL memory.
- Stdlib-first, no API calls in core logic; LLM/edge models live only in
  `verifier/` and `providers/`.
- Evidence is append-only forever; revision happens in the edge layer, never by
  deletion.
- Authorization is deterministic given the verified-edge graph. Verifiers may
  only propose or verify local edges; they may not decide final memory fate.
- Verifiers never call DPA; DPA never calls verifiers.

## 1. Locked amendments

- A1. `SUPERSEDES` carries a replacement. `EvidenceEdge` for `SUPERSEDES`
  records both the defeated belief and the replacement belief id; the
  resulting `DefeatPath` exposes `replacement_belief_id` so the authorized
  basis can immediately surface the current fact.
- A2. `CONDITION` is deleted as a relation label. Conditional semantics live
  on `DependencyEdge: REQUIRES(belief, condition)`; current unsatisfaction is
  expressed by `EvidenceEdge: BLOCKS(condition)`. `RELEASES` is reserved
  strictly for clearing a prior blocker.
- A3. `REAFFIRMS(belief)` is added as an evidence edge type so belief-level
  uncertainty can be cleared without abusing `RELEASES`.
- A4. Two retriever interfaces are introduced and kept separate:
  `ImpactCandidateRetriever` (write-time, evidence -> affected
  beliefs/conditions) and `QueryBeliefRetriever` (query-time, query ->
  answer-relevant beliefs). Neither may drop its primary input.
- A5. No `NO_DEFEAT` enum value. `AuthorizationTrace.accepted_defeat_path`
  is `DefeatPath | None`; an authorized belief has `None` and records its
  supporting evidence ids in a separate field.
- A6. No runtime compatibility shim. Old types
  (`EpisodicEvidence`, `Belief`, `RelationPrediction`,
  `AuthorizationDecision`, `EvaluationRecord`, `RelationType`,
  `BeliefStatus`, and the existing Phase-1 records) remain in `schemas.py`
  for one transition wave so current tests keep importing, but new runtime
  logic imports only the canonical typed graph schemas. The new typed
  graph is never converted back to `RelationPrediction`.
- A6.1. Two legacy enum *values* are retired in Wave 0 rather than
  preserved: `AuthorizationStatus.CONDITIONAL` (subsumed by the
  `BLOCKED` / `UNRESOLVED` split per A2) and `DefeatPathType.ROLLBACK_RELEASE`
  (replaced by `UNRESOLVED_UNCERTAIN` per A3 and A5, since release of a
  blocker is not itself a defeat outcome). A case-sensitive repository
  search across `src/`, `tests/`, and `scripts/` for the literal
  identifiers `CONDITIONAL` and `ROLLBACK_RELEASE` returned matches only
  inside the new amendment-A2 guard tests in
  `tests/test_schema_roundtrip.py`. No active runtime dependency on
  either removed value was found; the retirement is therefore taken at
  Wave 0 instead of being deferred. Wave 0 does not claim "no runtime
  behavior changed"; it claims "canonical schema semantics were
  introduced and no active runtime dependency on removed legacy enum
  members was found by repository search."
- A7. `condition_id` is scoped. `ConditionNode` carries an explicit
  `scope_id` (typically `user_id`); identical strings across users do not
  merge.
- A8. `DependencyEdge` has explicit provenance fields: `inducer`,
  `supporting_evidence_ids`, `model_call_trace_id`, `confidence`,
  `rationale`. These are first-class, not buried in metadata.
- A9. Test layering. Three suites must pass on every commit:
  `tests/gate_unit/`, `tests/verifier_contract/`,
  `tests/backend_contract/`. Research metrics live in
  `scripts/run_boundary_audit_dev.py` and report failure as a research
  signal, not as a CI failure.
- A10. `DirectJudgeBackend` and Phase 6 (local edge classifier / SFT / LoRA)
  are deferred until Waves 1-3 land and the typed path-based core is
  honest.

## 2. Refactor in six sequential steps

The current code mixes belief-internal dependency edges (`REQUIRED_BY`) with
evidence-update edges (`BLOCK`, `SUPERSEDE`, `UNCERTAIN`) in a single flat
`RelationPrediction` list. The whole refactor is structural: split those two
edge classes, then derive everything else from that split.

1. New graph primitives. Add `ConditionNode`, `DependencyEdge`,
   `EvidenceEdge` (with `replacement_belief_id`), `DefeatPath`,
   `AuthorizationTrace` as the canonical types in `schemas.py`. Keep the
   legacy types in-tree per A6 so existing tests still import.
2. Typed graph store. Rewrite `retracemem/memory/belief_store.py` to maintain
   four indexed collections (`beliefs`, `conditions`, `dependency_edges`,
   `evidence_edges`) instead of one `_relations` list, plus the helpers
   `dependencies_of(belief_id)`,
   `evidence_edges_for_condition(condition_id)`, and
   `evidence_edges_for_belief(belief_id)`.
3. Two verifier interfaces.
   - `RequirementInducer.induce_requirements(belief, evidence_context) -> list[RequirementProposal]`.
   - `EvidenceEdgeVerifier.verify_edges(new_evidence, candidate_belief, candidate_replacement_beliefs, candidate_conditions, temporal_context) -> list[EvidenceEdge]`.
   Note: During edge verification, `SUPERSEDES` edges must reference a real candidate replacement belief that is grounded in the new evidence (i.e. extraction of candidate new beliefs from new evidence must precede supersession verification). Heuristic implementations of these interfaces serve only as offline contract validation fixtures and must not be used in paper main-result runners.
4. Path-admitting gate. Rewrite `retracemem/tms/gate.py` so an
   `EvidenceEdge` is admitted only when:
   - `BLOCKS(e, c)` requires that some accepted
     `DependencyEdge: REQUIRES(b, c)` already exists for at least one
     belief `b`.
   - `RELEASES(e, c)` requires the same anchor.
   - `SUPERSEDES(e, b)` requires a non-null `replacement_belief_id`
     (per A1).
   - `REAFFIRMS(e, b)` requires that `b` exists.
   - `UNCERTAIN(e, b)` is admitted as a status edge but never alone causes
     `BLOCKED`; it produces only `UNRESOLVED`.
5. `DefeatPathAuthorizationAlgorithm`. Replace
   `retracemem/tms/authorization.py:AuthorizationEngine` with a function
   whose only inputs are the typed graph and `(belief, as_of_time |
   as_of_evidence_id)`. Returns `AuthorizationTrace`. No mock-ledger
   fallback; the current `_ensure_mock_ledger` path at
   `retracemem/tms/authorization.py:26-44` is deleted.
6. Query-conditioned basis. Rewrite
   `retracemem/generation/basis_builder.py` and
   `retracemem/backends/retrace_backend.py:search` to (a) retrieve
   query-relevant candidate beliefs first via `QueryBeliefRetriever`,
   (b) run DPA only over the retrieved set, (c) attach provenance and the
   accepted defeat path to each emitted item. The current `del query`
   lines are deleted.

## 3. Revised data schemas

```python
# retracemem.schemas (canonical typed graph schemas, additive in Wave 0).

@dataclass(frozen=True)
class EvidenceNode:
    evidence_id: str
    session_id: str
    timestamp: str | None
    text: str
    source_dataset: str
    source_pointer: str
    is_raw_source: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BeliefNode:
    belief_id: str
    proposition: str
    source_evidence_ids: tuple[str, ...] = ()
    source_span: str | None = None
    extractor_version: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionNode:
    condition_id: str
    scope_id: str        # user_id or store-scope namespace per A7
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DependencyEdge:
    edge_id: str
    belief_id: str
    condition_id: str
    inducer: str         # heuristic | prompt | manual_fixture | local_classifier
    edge_type: str = "REQUIRES"
    supporting_evidence_ids: tuple[str, ...] = ()
    model_call_trace_id: str | None = None
    confidence: float | None = None
    rationale: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EvidenceEdgeType(str, Enum):
    BLOCKS = "BLOCKS"           # target_kind = "condition"
    RELEASES = "RELEASES"       # target_kind = "condition"
    SUPERSEDES = "SUPERSEDES"   # target_kind = "belief"; replacement_belief_id required
    REAFFIRMS = "REAFFIRMS"     # target_kind = "belief"
    UNCERTAIN = "UNCERTAIN"     # target_kind = "belief"


@dataclass(frozen=True)
class EvidenceEdge:
    edge_id: str
    edge_type: EvidenceEdgeType
    evidence_id: str
    target_kind: str            # "condition" | "belief"
    target_id: str
    verifier: str               # heuristic | prompt | local_classifier
    replacement_belief_id: str | None = None     # required when SUPERSEDES
    valid_from: str | None = None
    valid_until: str | None = None
    confidence: float | None = None
    rationale: str | None = None
    span: str | None = None
    model_call_trace_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DefeatPathType(str, Enum):
    DIRECT_SUPERSEDE = "DIRECT_SUPERSEDE"
    PREREQUISITE_BLOCK = "PREREQUISITE_BLOCK"
    UNRESOLVED_UNCERTAIN = "UNRESOLVED_UNCERTAIN"


@dataclass(frozen=True)
class DefeatPath:
    path_id: str
    path_type: DefeatPathType
    target_belief_id: str
    supporting_dependency_edge_ids: tuple[str, ...] = ()
    supporting_evidence_edge_ids: tuple[str, ...] = ()
    replacement_belief_id: str | None = None     # populated for DIRECT_SUPERSEDE
    as_of_time: str | None = None
    as_of_evidence_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AuthorizationStatus(str, Enum):
    AUTHORIZED = "AUTHORIZED"
    BLOCKED = "BLOCKED"
    SUPERSEDED = "SUPERSEDED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class AuthorizationTrace:
    trace_id: str
    belief_id: str
    status: AuthorizationStatus
    accepted_defeat_path: DefeatPath | None = None     # None when AUTHORIZED, per A5
    considered_defeat_paths: tuple[DefeatPath, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    query_id: str | None = None
    as_of_time: str | None = None
    as_of_evidence_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

## 4. DefeatPathAuthorizationAlgorithm pseudocode

```text
function DPA(belief b, store M, as_of_time t, as_of_ev_id e_cut):
    # M = (E, B, C, D, U): evidence ledger, beliefs, conditions,
    # dependency edges, evidence edges. All edges have been admitted by
    # RevisionGate.

    # 1. Time-bounded view of evidence edges
    U_t = filter U by edge_valid_at(edge, t, e_cut)

    # 2. Direct supersession (highest precedence)
    super_edges = [u in U_t : u.edge_type == SUPERSEDES and u.target_id == b.id]
    if super_edges:
        latest = argmax_by_evidence_time(super_edges)
        return Trace(
            b, SUPERSEDED,
            accepted_path=DefeatPath(
                type=DIRECT_SUPERSEDE,
                target=b.id,
                evidence_edges=[latest.edge_id],
                dependency_edges=[],
                replacement_belief_id=latest.replacement_belief_id,  # per A1
            ),
        )

    # 3. Conditional defeat through any required condition
    deps = [d in D : d.belief_id == b.id]
    blocking_paths = []
    for d in deps:
        c = d.condition_id
        cond_edges = [u in U_t : u.target_kind == "condition" and u.target_id == c]
        if not cond_edges:
            continue
        latest_for_c = argmax_by_evidence_time(cond_edges)
        if latest_for_c.edge_type == BLOCKS:
            blocking_paths.append(DefeatPath(
                type=PREREQUISITE_BLOCK,
                target=b.id,
                dependency_edges=[d.edge_id],
                evidence_edges=[latest_for_c.edge_id],
            ))
        # RELEASES on c clears the blocker but does not assert b is currently
        # true; it only allows DPA to find no current blocker.

    if blocking_paths:
        chosen = max(blocking_paths, key=path_recency_key)
        return Trace(b, BLOCKED, accepted_path=chosen, considered=blocking_paths)

    # 4. Belief-level uncertainty, possibly cleared by a later REAFFIRMS
    belief_status_edges = [u in U_t :
                           u.target_id == b.id and
                           u.edge_type in (UNCERTAIN, REAFFIRMS)]
    if belief_status_edges:
        latest = argmax_by_evidence_time(belief_status_edges)
        if latest.edge_type == UNCERTAIN:
            return Trace(b, UNRESOLVED, accepted_path=DefeatPath(
                type=UNRESOLVED_UNCERTAIN,
                target=b.id,
                evidence_edges=[latest.edge_id],
            ))
        # latest is REAFFIRMS: fall through to AUTHORIZED.

    # 5. No admitted defeat path -> authorized
    return Trace(
        b, AUTHORIZED,
        accepted_path=None,                       # per A5
        supporting_evidence_ids=tuple(b.source_evidence_ids),
    )


function BuildAuthorizedBasis(query q, store M, t, e_cut, k):
    candidates = QueryBeliefRetriever(q, M.beliefs, k)        # uses q
    basis = []
    for b in candidates:
        trace = DPA(b, M, t, e_cut)
        if trace.status == AUTHORIZED:
            basis.append({
                belief: b,
                provenance: trace.supporting_evidence_ids,
                trace: trace,
            })
        elif trace.status == SUPERSEDED and
             trace.accepted_defeat_path.replacement_belief_id is not None:
            # Surface the replacement, not the defeated belief.
            r = M.beliefs[trace.accepted_defeat_path.replacement_belief_id]
            r_trace = DPA(r, M, t, e_cut)
            if r_trace.status == AUTHORIZED:
                basis.append({belief: r, provenance: ..., trace: r_trace})
    return basis
```

Invariants enforced by DPA:

- A `BLOCKS(e, c)` edge with no `REQUIRES(b, c)` for the belief contributes
  nothing.
- Releasing a condition does not assert truth of any belief; if a later
  `SUPERSEDES(b)` exists it still wins.
- `REAFFIRMS` clears prior `UNCERTAIN` only when it is strictly more
  recent.
- Determinism: tie-breaks by
  `(evidence_timestamp, ledger_index, edge_id)`.

## 5. Wave and subagent dispatch

Lead integration rules:

- Each wave commits separately and is reviewed before the next dispatch.
- No two subagents may write the same file in the same wave.
- No external API calls in core unit-test work.
- No official STALE / Memora performance evaluation during the refactor.

### Wave 0 - Lead only

Owned files:

- `docs/refactor_plan_defeat_path.md`
- `src/retracemem/schemas.py`
- `tests/test_schema_roundtrip.py`

Deliverables:

- locked plan with amendments A1-A10;
- canonical typed graph schemas additive to legacy types;
- round-trip tests for every new dataclass.

No runtime integration in Wave 0.

### Wave 1 - Subagents A and B in parallel

Subagent A: typed graph and deterministic DPA core.

Owned files:

- `src/retracemem/memory/belief_store.py`
- `src/retracemem/memory/temporal_validity.py`
- `src/retracemem/tms/gate.py`
- `src/retracemem/tms/authorization.py`
- `src/retracemem/tms/rollback.py`
- `tests/gate_unit/**`

Required gate-unit cases:

1. direct supersession with replacement belief surfaced;
2. prerequisite block;
3. unrelated belief preservation;
4. release of blocker followed by no later supersession;
5. release followed by a later supersession (supersession still wins);
6. uncertainty followed by reaffirmation (becomes authorized);
7. `BLOCKS` without a `REQUIRES` anchor is rejected by the gate.

Subagent B: verifier contracts and honest fixture infrastructure.

Owned files:

- `src/retracemem/verifier/**`
- `src/retracemem/extraction/**`
- `tests/verifier_contract/**`
- `scripts/record_verifier_cassettes.py`

Tasks:

- replace flat `RelationVerifier` with `RequirementInducer` and
  `EvidenceEdgeVerifier`;
- introduce manual and heuristic implementations for deterministic
  development;
- rename prompt verifier conceptually to `PromptEvidenceEdgeVerifier`;
- verifier tests must not be fed expected labels;
- do not use STALE or Memora official evaluation examples in test
  fixtures.

### Wave 2 - Subagent C

Subagent C: query-conditioned integration and smoke-run cleanup.

Owned files:

- `src/retracemem/retrieval/**`
- `src/retracemem/generation/basis_builder.py`
- `src/retracemem/backends/retrace_backend.py`
- `src/retracemem/pipeline.py`
- `scripts/run_retrace_internal_dev.py`
- existing STALE / Memora runner scripts
- `tests/backend_contract/**`

Tasks:

- split `ImpactCandidateRetriever` from `QueryBeliefRetriever`;
- ensure query-time basis genuinely uses the query;
- wire DPA over query-relevant candidate beliefs;
- attach provenance ids and optional `--expand-provenance` output;
- rename `run_stale_frozen_eval.py` and `run_memora_frozen_eval.py` to
  `run_stale_interface_smoke.py` and `run_memora_interface_smoke.py`,
  writing only to `outputs/smoke/`;
- replace label-injected internal-dev behavior with a research run that
  does not expose expected relation labels to the verifier.

### Wave 3 - Subagent D

Subagent D: documentation and contamination guards. Only after Wave 2.

Owned files:

- `docs/**` excluding the locked refactor plan unless the lead asks
- `tests/contamination/**`
- `registry/**`

Tasks:

- document benchmark separation: interface smoke, internal development,
  frozen official evaluation;
- add exact-copy contamination checks plus generation-provenance manifest
  records;
- explicitly state that exact-hash comparison catches exact copies only and
  is not proof against paraphrase contamination.

### Deferred

- `DirectJudgeBackend` (LLM-direct-judge attribution baseline).
- Phase 6 local-classifier / SFT / LoRA edge verifier.
- Any official STALE or Memora performance evaluation.

## 6. File inventory

### Preserve unchanged

- `retracemem/memory/episode_ledger.py`.
- `retracemem/cache/`, `retracemem/providers/`,
  `retracemem/evaluation/jsonl.py`, `cost_tracker.py`,
  `cost_accounting.py`, `records.py`.
- `retracemem/adapters/stale_adapter.py`,
  `retracemem/adapters/memora_adapter.py`.
- `data/boundary_audit/boundary_audit_dev.jsonl`,
  `data/boundary_audit/boundary_audit_eval.jsonl`.
- `tests/fixtures/toy_revision_cases.jsonl`.
- All of `reference/`.

### Modify

- `retracemem/schemas.py` (Wave 0).
- `retracemem/memory/belief_store.py`,
  `retracemem/memory/temporal_validity.py`,
  `retracemem/tms/gate.py`, `retracemem/tms/authorization.py`,
  `retracemem/tms/rollback.py` (Wave 1, Subagent A).
- `retracemem/verifier/`, `retracemem/extraction/` (Wave 1, Subagent B).
- `retracemem/retrieval/`, `retracemem/generation/basis_builder.py`,
  `retracemem/backends/retrace_backend.py`,
  `retracemem/pipeline.py` (Wave 2).

### Rename / split

- `retracemem/verifier/prompt_verifier.py`
  -> `retracemem/verifier/prompt_evidence_edge_verifier.py`.
- `retracemem/verifier/heuristic_verifier.py`
  -> `retracemem/verifier/heuristic_evidence_edge_verifier.py`.
- New: `retracemem/verifier/requirement_inducer.py`.
- `scripts/run_stale_frozen_eval.py`
  -> `scripts/run_stale_interface_smoke.py`.
- `scripts/run_memora_frozen_eval.py`
  -> `scripts/run_memora_interface_smoke.py`.
- `scripts/run_retrace_internal_dev.py`
  -> `scripts/run_gate_unit_replay.py` plus a new
  `scripts/run_boundary_audit_dev.py` for honest research metrics.

### Deprecate / quarantine

- `RelationType.REQUIRED_BY`: moves to `DependencyEdge` with
  `edge_type == "REQUIRES"`.
- `RelationType.CONDITION`: deleted per A2.
- The implicit ledger fabrication at
  `retracemem/tms/authorization.py:26-44` (`_ensure_mock_ledger`):
  deleted in Wave 1.
- `RetracePipeline._UncertainVerifier` fallback at
  `retracemem/pipeline.py:170`: deleted in Wave 2.
- `MockLLMProvider(default_response="{}")` branches in current
  `*_frozen_eval.py`: deleted in Wave 2.

## 7. Test migration plan

Existing tests conflate gate-unit, verifier, and end-to-end roles. Three
suites:

- `tests/gate_unit/` (Wave 1, Subagent A): deterministic injected-edge
  cases for DPA + gate. Must pass.
- `tests/verifier_contract/` (Wave 1, Subagent B): verifier interface and
  cassette tests. Verifier must not be fed expected labels. Must pass.
- `tests/backend_contract/` (Wave 2, Subagent C): query-conditioned basis,
  output schema, provenance hydration. Must pass.

Existing tests:

- `tests/test_tms_authorization.py`,
  `tests/test_pipeline.py` `StaticVerifier` cases ->
  migrate to `tests/gate_unit/`.
- `tests/test_prompt_verifier.py` -> migrate to
  `tests/verifier_contract/` minus mock-as-oracle cases.
- `tests/fixtures/toy_revision_cases.jsonl` ->
  `tests/gate_unit/fixtures/`.

Research runs:

- `scripts/run_boundary_audit_dev.py` reports
  `authorization_accuracy`, `unsupported_revision_rate`,
  `protected_belief_preservation`, `rollback_recovery`. Failures here are
  research signals; pytest does not gate them.

BoundaryAudit split:

- `boundary_audit_dev.jsonl` for verifier prompt and heuristic tuning.
- `boundary_audit_eval.jsonl` frozen; never inspected during tuning.
- Neither file may be derived from STALE / Memora official examples;
  enforced in `tests/contamination/` (Wave 3, Subagent D).

## 8. Confirmed decisions (orchestrator review, 2026-05-28)

1. Old-schema compatibility: legacy types remain in-tree for one wave; new
   runtime imports only canonical types; no full compat shim.
2. `DirectJudgeBackend`: deferred to a later wave.
3. `AuthorizationTrace` provenance shape: edge ids by default,
   `--expand-provenance` flag in runners.

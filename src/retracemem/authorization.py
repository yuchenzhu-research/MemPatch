from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retracemem.memory.belief_store import BeliefStore
from retracemem.memory.episode_ledger import EpisodeLedger
from retracemem.methods.contracts import SharedCandidateView
from retracemem.schemas import AuthorizationStatus, EvidenceEdge
from retracemem.tms.authorization import DefeatPathAuthorizationAlgorithm
from retracemem.tms.gate import RevisionGate


@dataclass(frozen=True)
class EvidenceProposalBatch:
    edges: tuple[EvidenceEdge, ...]
    model_call_trace_id: str | None = None
    source_belief_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuthorizationResult:
    authorized_belief_ids: tuple[str, ...]
    excluded_belief_ids: tuple[str, ...]
    trace: dict[str, Any]


_STATUS_MAP = {
    AuthorizationStatus.AUTHORIZED: "AUTHORIZED",
    AuthorizationStatus.BLOCKED: "BLOCKED",
    AuthorizationStatus.SUPERSEDED: "SUPERSEDED",
    AuthorizationStatus.UNRESOLVED: "UNRESOLVED",
}


def authorize(
    view: SharedCandidateView,
    proposal_batches: tuple[EvidenceProposalBatch, ...],
    *,
    bypass_gate: bool = False,
    audit_metadata: dict[str, Any] | None = None,
) -> AuthorizationResult:
    """Deterministic authorization kernel (sole public entrypoint).

    The model proposes typed revision patches; DPA authorizes; MemPatch-Bench
    evaluates the resulting ``memory_state``. Callers must not invoke
    :class:`RevisionGate` or the DPA directly — use this entrypoint only.

    Mapping to the MemPatch Revision Module (concept -> code):

    * ``G_t = (E_t, B_t, C_t, D_t, R_t)`` — the typed graph state assembled below
      from the method-visible ``view``:
        - ``E_t`` evidence ledger        -> ``view.evidence_context`` -> ``ledger``
        - ``B_t`` beliefs                -> ``view.candidate_beliefs`` (+ replacements)
        - ``C_t`` conditions             -> ``view.candidate_conditions_by_belief``
        - ``D_t`` REQUIRES anchors       -> ``view.dependency_edges_by_belief``
        - ``R_t`` admitted revision edges -> edges added to ``store`` after the gate
    * Typed actions ``A_t`` — the proposed ``EvidenceEdge`` objects carried in
      ``proposal_batches`` (SUPERSEDES/BLOCKS/RELEASES/REAFFIRMS/UNCERTAIN).
    * RevisionGate ``Gamma`` — ``gate.admit_*``: structural admission of proposed
      effects; rejected edges produce no graph mutation (fail-closed). With
      ``bypass_gate`` set (ablation only) admission is forced True.
    * Append-only graph update — admitted edges are *added*; nothing is deleted,
      preserving the immutable-evidence invariant.
    * DPA status ``sigma_t(b)`` — ``DefeatPathAuthorizationAlgorithm.authorize``
      computes each belief's status with canonical precedence
      ``SUPERSEDES > PREREQUISITE_BLOCK > UNRESOLVED_UNCERTAIN > AUTHORIZED`` and
      deterministic temporal tie-breaking; only ``AUTHORIZED`` beliefs enter
      ``authorized_belief_ids``.

    The returned ``trace`` records admitted anchors, per-edge gate decisions, and
    accepted defeat paths so the decision is fully reconstructable for audit.
    """
    ledger = EpisodeLedger()
    store = BeliefStore()
    gate = RevisionGate()

    for ev in view.evidence_context:
        ledger.append(ev)

    for belief in view.candidate_beliefs:
        store.add_belief(belief)
    for belief in view.candidate_replacement_beliefs:
        if not store.has_belief(belief.belief_id):
            store.add_belief(belief)

    for _bid, conditions in view.candidate_conditions_by_belief:
        for condition in conditions:
            if not store.has_condition(condition.condition_id):
                store.add_condition(condition)

    admitted_anchors: list[dict[str, Any]] = []
    for _bid, dependencies in view.dependency_edges_by_belief:
        for dependency in dependencies:
            decision = gate.admit_dependency_edge(dependency, store)
            if not decision.admitted:
                raise ValueError(
                    f"Fixed supplied DependencyEdge '{dependency.edge_id}' "
                    f"rejected by RevisionGate: {decision.reason}"
                )
            store.add_dependency_edge(dependency)
            admitted_anchors.append({
                "edge_id": dependency.edge_id,
                "belief_id": dependency.belief_id,
                "condition_id": dependency.condition_id,
            })

    trace_ids: list[str] = []
    edge_proposals: list[dict[str, Any]] = []
    for batch in proposal_batches:
        if batch.model_call_trace_id and batch.model_call_trace_id not in trace_ids:
            trace_ids.append(batch.model_call_trace_id)
        for edge in batch.edges:
            if bypass_gate:
                admitted = True
                reason = "Bypassed RevisionGate (Ablation)"
            else:
                decision = gate.admit_evidence_edge(edge, store)
                admitted = decision.admitted
                reason = decision.reason
            proposal = {
                "edge_id": edge.edge_id,
                "edge_type": edge.edge_type.value,
                "target_id": edge.target_id,
                "admitted": admitted,
                "gate_reason": reason,
                "model_call_trace_id": batch.model_call_trace_id,
            }
            if batch.source_belief_id is not None:
                proposal["belief_id"] = batch.source_belief_id
            edge_proposals.append(proposal)
            if admitted:
                store.add_evidence_edge(edge)

    dpa = DefeatPathAuthorizationAlgorithm(store, ledger)
    authorized_ids: list[str] = []
    excluded_ids: list[str] = []
    fine_grained: dict[str, str] = {}
    defeat_paths: list[dict[str, Any]] = []

    for belief in view.candidate_beliefs:
        trace = dpa.authorize(
            belief.belief_id,
            as_of_evidence_id=view.new_evidence.evidence_id,
            query_id=view.query_id,
        )
        status = _STATUS_MAP.get(trace.status, trace.status.value)
        fine_grained[belief.belief_id] = status

        if trace.status == AuthorizationStatus.AUTHORIZED:
            authorized_ids.append(belief.belief_id)
        else:
            excluded_ids.append(belief.belief_id)
            if trace.accepted_defeat_path is not None:
                defeat_paths.append({
                    "belief_id": belief.belief_id,
                    "path_type": trace.accepted_defeat_path.path_type.value,
                    "path_id": trace.accepted_defeat_path.path_id,
                    "evidence_edge_ids": list(trace.accepted_defeat_path.supporting_evidence_edge_ids),
                    "dependency_edge_ids": list(trace.accepted_defeat_path.supporting_dependency_edge_ids),
                    "replacement_belief_id": trace.accepted_defeat_path.replacement_belief_id,
                })

    provenance: dict[str, Any] = {
        "view_fingerprint": view.view_fingerprint,
        "fine_grained_statuses": fine_grained,
        "defeat_paths": defeat_paths,
        "admitted_fixed_anchors": admitted_anchors,
        "edge_proposals": edge_proposals,
        "model_call_trace_ids": tuple(trace_ids),
    }
    if audit_metadata:
        provenance.update(audit_metadata)

    return AuthorizationResult(
        authorized_belief_ids=tuple(authorized_ids),
        excluded_belief_ids=tuple(excluded_ids),
        trace=provenance,
    )

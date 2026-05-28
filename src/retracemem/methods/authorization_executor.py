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
class ProposedEvidenceEdges:
    edges: tuple[EvidenceEdge, ...]
    model_call_trace_id: str
    source_belief_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuthorizationExecutionResult:
    authorized_belief_ids: tuple[str, ...]
    excluded_belief_ids: tuple[str, ...]
    model_call_trace_ids: tuple[str, ...]
    provenance: dict[str, Any]


_STATUS_MAP = {
    AuthorizationStatus.AUTHORIZED: "AUTHORIZED",
    AuthorizationStatus.BLOCKED: "BLOCKED",
    AuthorizationStatus.SUPERSEDED: "SUPERSEDED",
    AuthorizationStatus.UNRESOLVED: "UNRESOLVED",
}


def execute_authorization(
    view: SharedCandidateView,
    proposal_batches: tuple[ProposedEvidenceEdges, ...],
    *,
    base_provenance: dict[str, Any] | None = None,
) -> AuthorizationExecutionResult:
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
        if batch.model_call_trace_id not in trace_ids:
            trace_ids.append(batch.model_call_trace_id)
        for edge in batch.edges:
            decision = gate.admit_evidence_edge(edge, store)
            proposal = {
                "edge_id": edge.edge_id,
                "edge_type": edge.edge_type.value,
                "target_id": edge.target_id,
                "admitted": decision.admitted,
                "gate_reason": decision.reason,
                "model_call_trace_id": batch.model_call_trace_id,
            }
            if batch.source_belief_id is not None:
                proposal["belief_id"] = batch.source_belief_id
            edge_proposals.append(proposal)
            if decision.admitted:
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
    }
    if base_provenance:
        provenance.update(base_provenance)

    return AuthorizationExecutionResult(
        authorized_belief_ids=tuple(authorized_ids),
        excluded_belief_ids=tuple(excluded_ids),
        model_call_trace_ids=tuple(trace_ids),
        provenance=provenance,
    )


def cost_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    tokens_before = before.get("tokens", {})
    tokens_after = after.get("tokens", {})
    calls_before = before.get("calls", {})
    calls_after = after.get("calls", {})
    return {
        "latency_ms": after.get("latency_ms", 0.0) - before.get("latency_ms", 0.0),
        "tokens": {
            "prompt": tokens_after.get("prompt", 0) - tokens_before.get("prompt", 0),
            "completion": tokens_after.get("completion", 0) - tokens_before.get("completion", 0),
            "total": tokens_after.get("total", 0) - tokens_before.get("total", 0),
        },
        "calls": {
            key: calls_after.get(key, 0) - calls_before.get(key, 0)
            for key in set(calls_after) | set(calls_before)
        },
        "cache_hits": after.get("cache_hits", 0) - before.get("cache_hits", 0),
        "cache_misses": after.get("cache_misses", 0) - before.get("cache_misses", 0),
    }

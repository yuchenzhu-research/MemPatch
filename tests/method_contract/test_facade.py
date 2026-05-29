from __future__ import annotations

import json
import pytest

from retracemem.methods.contracts import SharedCandidateView
from retracemem.methods.facade import (
    AuthorizationRequest,
    AuthorizationResult,
    AuthorizationFacade,
)
from retracemem.methods.authorization_executor import ProposedEvidenceEdges
from retracemem.schemas import (
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceNode,
    EvidenceEdge,
    EvidenceEdgeType,
)


def _make_evidence(eid: str = "ev_new", text: str = "User broke their leg.") -> EvidenceNode:
    return EvidenceNode(
        evidence_id=eid,
        session_id="s1",
        timestamp="2026-01-02T00:00:00Z",
        text=text,
        source_dataset="test",
        source_pointer="ptr",
    )


def _make_old_evidence() -> EvidenceNode:
    return EvidenceNode(
        evidence_id="ev_old",
        session_id="s1",
        timestamp="2026-01-01T00:00:00Z",
        text="User commutes by bicycle.",
        source_dataset="test",
        source_pointer="ptr",
    )


def _make_belief(bid: str = "b_bike") -> BeliefNode:
    return BeliefNode(
        belief_id=bid,
        proposition="The user commutes by bicycle.",
        source_evidence_ids=("ev_old",),
    )


def _make_condition(cid: str = "c_leg") -> ConditionNode:
    return ConditionNode(condition_id=cid, scope_id="user1", text="User is physically able.")


def _make_dep_edge(bid: str = "b_bike", cid: str = "c_leg") -> DependencyEdge:
    return DependencyEdge(
        edge_id=f"dep-{bid}-{cid}",
        belief_id=bid,
        condition_id=cid,
        inducer="test_fixture",
        edge_type="REQUIRES",
    )


def _make_view() -> SharedCandidateView:
    ev_old = _make_old_evidence()
    ev_new = _make_evidence()
    b1 = _make_belief("b_bike")
    c1 = _make_condition("c_leg")
    dep = _make_dep_edge("b_bike", "c_leg")
    return SharedCandidateView(
        instance_id="case_1",
        query_id="q_1",
        query="How does the user commute?",
        evidence_context=(ev_old, ev_new),
        candidate_beliefs=(b1,),
        candidate_replacement_beliefs=(),
        candidate_conditions_by_belief=(("b_bike", (c1,)),),
        dependency_edges_by_belief=(("b_bike", (dep,)),),
        new_evidence=ev_new,
    )


def test_facade_basic_authorization() -> None:
    view = _make_view()
    req = AuthorizationRequest(view=view, provenance={"source_system": "subagent_1"})

    # BLOCKS edge proposal
    proposed = ProposedEvidenceEdges(
        edges=(
            EvidenceEdge(
                edge_id="ev:ev_new:BLOCKS:c_leg",
                edge_type=EvidenceEdgeType.BLOCKS,
                evidence_id="ev_new",
                target_kind="condition",
                target_id="c_leg",
                verifier="test",
            ),
        ),
        model_call_trace_id="call_trace_123",
    )

    res = AuthorizationFacade.authorize(req, (proposed,))

    assert "b_bike" in res.excluded_belief_ids
    assert "b_bike" not in res.authorized_belief_ids
    assert res.fine_grained_statuses["b_bike"] == "BLOCKED"
    assert res.provenance == {"source_system": "subagent_1"}
    assert res.trace["source_system"] == "subagent_1"


def test_facade_provenance_invariance() -> None:
    view = _make_view()
    proposed = ProposedEvidenceEdges(
        edges=(
            EvidenceEdge(
                edge_id="ev:ev_new:BLOCKS:c_leg",
                edge_type=EvidenceEdgeType.BLOCKS,
                evidence_id="ev_new",
                target_kind="condition",
                target_id="c_leg",
                verifier="test",
            ),
        ),
        model_call_trace_id="call_trace_123",
    )

    req1 = AuthorizationRequest(view=view, provenance={"source_system": "sys_a", "run_id": "1"})
    req2 = AuthorizationRequest(view=view, provenance={"source_system": "sys_b", "run_id": "2"})

    res1 = AuthorizationFacade.authorize(req1, (proposed,))
    res2 = AuthorizationFacade.authorize(req2, (proposed,))

    # The metadata should not change DPA outcomes
    assert res1.fine_grained_statuses == res2.fine_grained_statuses
    assert res1.authorized_belief_ids == res2.authorized_belief_ids
    assert res1.excluded_belief_ids == res2.excluded_belief_ids


def test_facade_json_serializable() -> None:
    view = _make_view()
    proposed = ProposedEvidenceEdges(
        edges=(),
        model_call_trace_id="trace_empty",
    )
    req = AuthorizationRequest(view=view, provenance={"producer_kind": "agent"})
    res = AuthorizationFacade.authorize(req, (proposed,))

    # Verify serialization does not raise error
    serialized_trace = json.dumps(res.trace)
    assert "view_fingerprint" in serialized_trace
    assert "fine_grained_statuses" in serialized_trace

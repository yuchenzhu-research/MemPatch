"""Contract tests for SharedCandidateView and controlled-comparison contracts."""
from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from retracemem.methods.contracts import (
    ControlledMethodResult,
    DirectUsabilityStatus,
    DirectUsabilityVerdict,
    SharedCandidateView,
)
from retracemem.schemas import BeliefNode, ConditionNode, DependencyEdge, EvidenceNode


def _make_evidence(eid: str = "ev1") -> EvidenceNode:
    return EvidenceNode(
        evidence_id=eid,
        session_id="s1",
        timestamp="2026-01-01T00:00:00Z",
        text="User broke their leg.",
        source_dataset="test",
        source_pointer="ptr",
    )


def _make_belief(bid: str = "b1", eid: str = "ev1") -> BeliefNode:
    return BeliefNode(
        belief_id=bid,
        proposition="The user commutes by bicycle.",
        source_evidence_ids=(eid,),
    )


def _make_condition(cid: str = "c1") -> ConditionNode:
    return ConditionNode(condition_id=cid, scope_id="user1", text="User is physically able.")


def _make_dep_edge(bid: str = "b_bike", cid: str = "c_leg") -> DependencyEdge:
    return DependencyEdge(
        edge_id=f"dep-{bid}-{cid}",
        belief_id=bid,
        condition_id=cid,
        inducer="test",
        edge_type="REQUIRES",
    )


def _make_view() -> SharedCandidateView:
    ev = _make_evidence()
    b1 = _make_belief("b_bike", "ev1")
    b_new = _make_belief("b_car", "ev1")
    c1 = _make_condition("c_leg")
    dep = _make_dep_edge("b_bike", "c_leg")
    return SharedCandidateView(
        instance_id="case_1",
        query_id="q_1",
        query="How does the user commute?",
        evidence_context=(ev,),
        candidate_beliefs=(b1,),
        candidate_replacement_beliefs=(b_new,),
        candidate_conditions_by_belief=(("b_bike", (c1,)),),
        dependency_edges_by_belief=(("b_bike", (dep,)),),
        new_evidence=ev,
    )


def test_shared_candidate_view_preserves_identical_inputs() -> None:
    view = _make_view()
    view_copy = SharedCandidateView(
        instance_id=view.instance_id,
        query_id=view.query_id,
        query=view.query,
        evidence_context=view.evidence_context,
        candidate_beliefs=view.candidate_beliefs,
        candidate_replacement_beliefs=view.candidate_replacement_beliefs,
        candidate_conditions_by_belief=view.candidate_conditions_by_belief,
        dependency_edges_by_belief=view.dependency_edges_by_belief,
        new_evidence=view.new_evidence,
        metadata=view.metadata,
    )
    assert view.instance_id == view_copy.instance_id
    assert view.candidate_beliefs == view_copy.candidate_beliefs
    assert view.candidate_replacement_beliefs == view_copy.candidate_replacement_beliefs
    assert view.evidence_context == view_copy.evidence_context
    assert view.candidate_conditions_by_belief == view_copy.candidate_conditions_by_belief
    assert view.view_fingerprint == view_copy.view_fingerprint


def test_fingerprint_deterministic() -> None:
    v1 = _make_view()
    v2 = _make_view()
    assert v1.view_fingerprint == v2.view_fingerprint
    assert len(v1.view_fingerprint) == 64  # sha256 hex


def test_fingerprint_changes_with_input() -> None:
    v1 = _make_view()
    ev = _make_evidence("ev2")
    b1 = _make_belief("b_bike", "ev1")
    b_new = _make_belief("b_car", "ev1")
    c1 = _make_condition("c_leg")
    dep = _make_dep_edge("b_bike", "c_leg")
    v2 = SharedCandidateView(
        instance_id="case_2",
        query_id="q_1",
        query="How does the user commute?",
        evidence_context=(_make_evidence(), ev),
        candidate_beliefs=(b1,),
        candidate_replacement_beliefs=(b_new,),
        candidate_conditions_by_belief=(("b_bike", (c1,)),),
        dependency_edges_by_belief=(("b_bike", (dep,)),),
        new_evidence=ev,
    )
    assert v1.view_fingerprint != v2.view_fingerprint


def test_new_evidence_must_be_in_context() -> None:
    ev1 = _make_evidence("ev1")
    ev2 = _make_evidence("ev2")
    b1 = _make_belief("b1", "ev1")
    with pytest.raises(ValueError, match="must appear in evidence_context"):
        SharedCandidateView(
            instance_id="x", query_id="q", query="q",
            evidence_context=(ev1,),
            candidate_beliefs=(b1,),
            candidate_replacement_beliefs=(),
            new_evidence=ev2,
        )


def test_direct_usability_status_roundtrip() -> None:
    for status in DirectUsabilityStatus:
        assert DirectUsabilityStatus(status.value) is status
    raw = json.dumps(DirectUsabilityStatus.USABLE.value)
    assert json.loads(raw) == "USABLE"


def test_direct_usability_verdict_has_no_edge_semantics() -> None:
    verdict = DirectUsabilityVerdict(
        belief_id="b1",
        status=DirectUsabilityStatus.NOT_USABLE,
        rationale="Belief is outdated by new evidence.",
        model_call_trace_id="trace-1",
        confidence=0.95,
    )
    d = asdict(verdict)
    assert "edge_type" not in d
    assert "target_kind" not in d
    assert "replacement_belief_id" not in d
    assert "path_type" not in d
    assert d["status"] == "NOT_USABLE"
    assert d["rationale"] == "Belief is outdated by new evidence."


def test_controlled_method_result_fields() -> None:
    result = ControlledMethodResult(
        method_name="retrace_llm",
        instance_id="case_1",
        query_id="q_1",
        authorized_belief_ids=("b_car",),
        excluded_belief_ids=("b_bike",),
        model_call_trace_ids=("trace-1", "trace-2"),
        cost={"tokens": {"total": 150}},
    )
    assert result.method_name == "retrace_llm"
    assert "b_bike" in result.excluded_belief_ids
    assert len(result.model_call_trace_ids) == 2


def test_duplicate_candidate_belief_ids_rejected() -> None:
    ev = _make_evidence()
    b1 = _make_belief("b_dup", "ev1")
    b2 = _make_belief("b_dup", "ev1")
    with pytest.raises(ValueError, match="duplicate belief_ids"):
        SharedCandidateView(
            instance_id="x", query_id="q", query="q",
            evidence_context=(ev,),
            new_evidence=ev,
            candidate_beliefs=(b1, b2),
            candidate_replacement_beliefs=(),
        )


def test_duplicate_replacement_belief_ids_rejected() -> None:
    ev = _make_evidence()
    b1 = _make_belief("b1", "ev1")
    r1 = _make_belief("r_dup", "ev1")
    r2 = _make_belief("r_dup", "ev1")
    with pytest.raises(ValueError, match="duplicate belief_ids"):
        SharedCandidateView(
            instance_id="x", query_id="q", query="q",
            evidence_context=(ev,),
            new_evidence=ev,
            candidate_beliefs=(b1,),
            candidate_replacement_beliefs=(r1, r2),
        )


def test_invalid_conditions_key_rejected() -> None:
    ev = _make_evidence()
    b1 = _make_belief("b1", "ev1")
    c1 = _make_condition("c1")
    with pytest.raises(ValueError, match="not a candidate belief id"):
        SharedCandidateView(
            instance_id="x", query_id="q", query="q",
            evidence_context=(ev,),
            new_evidence=ev,
            candidate_beliefs=(b1,),
            candidate_replacement_beliefs=(),
            candidate_conditions_by_belief=(("nonexistent", (c1,)),),
        )


def test_duplicate_condition_ids_in_belief_rejected() -> None:
    ev = _make_evidence()
    b1 = _make_belief("b1", "ev1")
    c1 = _make_condition("c_dup")
    c2 = _make_condition("c_dup")
    with pytest.raises(ValueError, match="Duplicate condition_ids"):
        SharedCandidateView(
            instance_id="x", query_id="q", query="q",
            evidence_context=(ev,),
            new_evidence=ev,
            candidate_beliefs=(b1,),
            candidate_replacement_beliefs=(),
            candidate_conditions_by_belief=(("b1", (c1, c2)),),
        )


def test_dependency_edge_wrong_type_rejected() -> None:
    ev = _make_evidence()
    b1 = _make_belief("b1", "ev1")
    c1 = _make_condition("c1")
    bad_dep = DependencyEdge(
        edge_id="dep1", belief_id="b1", condition_id="c1",
        inducer="test", edge_type="BLOCKS",
    )
    with pytest.raises(ValueError, match="only 'REQUIRES' is permitted"):
        SharedCandidateView(
            instance_id="x", query_id="q", query="q",
            evidence_context=(ev,),
            new_evidence=ev,
            candidate_beliefs=(b1,),
            candidate_replacement_beliefs=(),
            candidate_conditions_by_belief=(("b1", (c1,)),),
            dependency_edges_by_belief=(("b1", (bad_dep,)),),
        )


def test_dependency_edge_dangling_condition_rejected() -> None:
    ev = _make_evidence()
    b1 = _make_belief("b1", "ev1")
    c1 = _make_condition("c1")
    dep = DependencyEdge(
        edge_id="dep1", belief_id="b1", condition_id="c_missing",
        inducer="test", edge_type="REQUIRES",
    )
    with pytest.raises(ValueError, match="not supplied for belief"):
        SharedCandidateView(
            instance_id="x", query_id="q", query="q",
            evidence_context=(ev,),
            new_evidence=ev,
            candidate_beliefs=(b1,),
            candidate_replacement_beliefs=(),
            candidate_conditions_by_belief=(("b1", (c1,)),),
            dependency_edges_by_belief=(("b1", (dep,)),),
        )


def test_dependency_edge_wrong_belief_rejected() -> None:
    ev = _make_evidence()
    b1 = _make_belief("b1", "ev1")
    c1 = _make_condition("c1")
    dep = DependencyEdge(
        edge_id="dep1", belief_id="b_other", condition_id="c1",
        inducer="test", edge_type="REQUIRES",
    )
    with pytest.raises(ValueError, match="targets belief"):
        SharedCandidateView(
            instance_id="x", query_id="q", query="q",
            evidence_context=(ev,),
            new_evidence=ev,
            candidate_beliefs=(b1,),
            candidate_replacement_beliefs=(),
            candidate_conditions_by_belief=(("b1", (c1,)),),
            dependency_edges_by_belief=(("b1", (dep,)),),
        )


def test_duplicate_dependency_edge_ids_rejected() -> None:
    ev = _make_evidence()
    b1 = _make_belief("b1", "ev1")
    c1 = _make_condition("c1")
    c2 = _make_condition("c2")
    dep1 = DependencyEdge(
        edge_id="dep_same", belief_id="b1", condition_id="c1",
        inducer="test", edge_type="REQUIRES",
    )
    dep2 = DependencyEdge(
        edge_id="dep_same", belief_id="b1", condition_id="c2",
        inducer="test", edge_type="REQUIRES",
    )
    with pytest.raises(ValueError, match="Duplicate dependency edge_id"):
        SharedCandidateView(
            instance_id="x", query_id="q", query="q",
            evidence_context=(ev,),
            new_evidence=ev,
            candidate_beliefs=(b1,),
            candidate_replacement_beliefs=(),
            candidate_conditions_by_belief=(("b1", (c1, c2)),),
            dependency_edges_by_belief=(("b1", (dep1, dep2)),),
        )


def test_overlap_candidate_and_replacement_rejected() -> None:
    ev = _make_evidence()
    b1 = _make_belief("b_same", "ev1")
    r1 = _make_belief("b_same", "ev1")
    with pytest.raises(ValueError, match="overlap"):
        SharedCandidateView(
            instance_id="x", query_id="q", query="q",
            evidence_context=(ev,),
            new_evidence=ev,
            candidate_beliefs=(b1,),
            candidate_replacement_beliefs=(r1,),
        )


def test_duplicate_evidence_ids_in_context_rejected() -> None:
    ev = _make_evidence("ev_dup")
    ev2 = _make_evidence("ev_dup")
    b1 = _make_belief("b1", "ev_dup")
    with pytest.raises(ValueError, match="duplicate evidence_ids"):
        SharedCandidateView(
            instance_id="x", query_id="q", query="q",
            evidence_context=(ev, ev2),
            new_evidence=ev,
            candidate_beliefs=(b1,),
            candidate_replacement_beliefs=(),
        )


def test_new_evidence_payload_mismatch_rejected() -> None:
    ev1 = EvidenceNode(
        evidence_id="ev1", session_id="s1", timestamp="2026-01-01T00:00:00Z",
        text="Text A", source_dataset="test", source_pointer="ptr",
    )
    ev1_diff = EvidenceNode(
        evidence_id="ev1", session_id="s1", timestamp="2026-01-01T00:00:00Z",
        text="Text B", source_dataset="test", source_pointer="ptr",
    )
    b1 = _make_belief("b1", "ev1")
    with pytest.raises(ValueError, match="different payload"):
        SharedCandidateView(
            instance_id="x", query_id="q", query="q",
            evidence_context=(ev1,),
            new_evidence=ev1_diff,
            candidate_beliefs=(b1,),
            candidate_replacement_beliefs=(),
        )


def test_repeated_condition_key_rejected() -> None:
    ev = _make_evidence()
    b1 = _make_belief("b1", "ev1")
    c1 = _make_condition("c1")
    with pytest.raises(ValueError, match="repeated key"):
        SharedCandidateView(
            instance_id="x", query_id="q", query="q",
            evidence_context=(ev,),
            new_evidence=ev,
            candidate_beliefs=(b1,),
            candidate_replacement_beliefs=(),
            candidate_conditions_by_belief=(("b1", (c1,)), ("b1", (c1,))),
        )


def test_repeated_dependency_key_rejected() -> None:
    ev = _make_evidence()
    b1 = _make_belief("b1", "ev1")
    c1 = _make_condition("c1")
    dep = DependencyEdge(
        edge_id="dep1", belief_id="b1", condition_id="c1",
        inducer="test", edge_type="REQUIRES",
    )
    with pytest.raises(ValueError, match="repeated key"):
        SharedCandidateView(
            instance_id="x", query_id="q", query="q",
            evidence_context=(ev,),
            new_evidence=ev,
            candidate_beliefs=(b1,),
            candidate_replacement_beliefs=(),
            candidate_conditions_by_belief=(("b1", (c1,)),),
            dependency_edges_by_belief=(("b1", (dep,)), ("b1", (dep,))),
        )


def test_conflicting_condition_payloads_rejected() -> None:
    ev = _make_evidence()
    b1 = _make_belief("b1", "ev1")
    b2 = _make_belief("b2", "ev1")
    c1a = ConditionNode(condition_id="c_shared", scope_id="u1", text="Text A")
    c1b = ConditionNode(condition_id="c_shared", scope_id="u1", text="Text B")
    with pytest.raises(ValueError, match="conflicting payloads"):
        SharedCandidateView(
            instance_id="x", query_id="q", query="q",
            evidence_context=(ev,),
            new_evidence=ev,
            candidate_beliefs=(b1, b2),
            candidate_replacement_beliefs=(),
            candidate_conditions_by_belief=(("b1", (c1a,)), ("b2", (c1b,))),
        )


def test_metadata_does_not_affect_fingerprint() -> None:
    """Metadata is non-semantic: changing it must not change the fingerprint.
    Controlled execution MUST NOT read metadata.
    """
    ev = _make_evidence()
    b1 = _make_belief("b_bike", "ev1")
    b_car = _make_belief("b_car", "ev1")
    v1 = SharedCandidateView(
        instance_id="case_1", query_id="q_1", query="q",
        evidence_context=(ev,), new_evidence=ev,
        candidate_beliefs=(b1,),
        candidate_replacement_beliefs=(b_car,),
        metadata={"debug": "value_1"},
    )
    v2 = SharedCandidateView(
        instance_id="case_1", query_id="q_1", query="q",
        evidence_context=(ev,), new_evidence=ev,
        candidate_beliefs=(b1,),
        candidate_replacement_beliefs=(b_car,),
        metadata={"debug": "value_2"},
    )
    assert v1.view_fingerprint == v2.view_fingerprint


def test_view_fingerprint_is_not_caller_settable() -> None:
    """view_fingerprint is init=False; callers cannot supply it."""
    ev = _make_evidence()
    b1 = _make_belief("b1", "ev1")
    with pytest.raises(TypeError):
        SharedCandidateView(
            instance_id="x", query_id="q", query="q",
            evidence_context=(ev,), new_evidence=ev,
            candidate_beliefs=(b1,),
            candidate_replacement_beliefs=(),
            view_fingerprint="fake",
        )


def test_contracts_do_not_import_legacy_types() -> None:
    import retracemem.methods.contracts as mod

    source = open(mod.__file__).read()
    assert "RelationPrediction" not in source
    assert "RelationType" not in source
    assert "EpisodicEvidence" not in source

"""Contract tests for SharedCandidateView and controlled-comparison contracts."""
from __future__ import annotations

import json
from dataclasses import asdict

from retracemem.methods.contracts import (
    ControlledMethodResult,
    DirectUsabilityStatus,
    DirectUsabilityVerdict,
    SharedCandidateView,
)
from retracemem.schemas import BeliefNode, ConditionNode, EvidenceNode


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


def _make_view() -> SharedCandidateView:
    ev = _make_evidence()
    b1 = _make_belief("b_bike", "ev1")
    b_new = _make_belief("b_car", "ev1")
    c1 = _make_condition("c_leg")
    return SharedCandidateView(
        instance_id="case_1",
        query_id="q_1",
        query="How does the user commute?",
        evidence_context=(ev,),
        candidate_beliefs=(b1,),
        candidate_replacement_beliefs=(b_new,),
        candidate_conditions_by_belief={"b_bike": (c1,)},
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
        metadata=view.metadata,
    )
    assert view.instance_id == view_copy.instance_id
    assert view.candidate_beliefs == view_copy.candidate_beliefs
    assert view.candidate_replacement_beliefs == view_copy.candidate_replacement_beliefs
    assert view.evidence_context == view_copy.evidence_context
    assert view.candidate_conditions_by_belief == view_copy.candidate_conditions_by_belief


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
    import pytest
    ev = _make_evidence()
    b1 = _make_belief("b_dup", "ev1")
    b2 = _make_belief("b_dup", "ev1")
    with pytest.raises(ValueError, match="duplicate belief_ids"):
        SharedCandidateView(
            instance_id="x", query_id="q", query="q",
            evidence_context=(ev,),
            candidate_beliefs=(b1, b2),
            candidate_replacement_beliefs=(),
        )


def test_duplicate_replacement_belief_ids_rejected() -> None:
    import pytest
    ev = _make_evidence()
    b1 = _make_belief("b1", "ev1")
    r1 = _make_belief("r_dup", "ev1")
    r2 = _make_belief("r_dup", "ev1")
    with pytest.raises(ValueError, match="duplicate belief_ids"):
        SharedCandidateView(
            instance_id="x", query_id="q", query="q",
            evidence_context=(ev,),
            candidate_beliefs=(b1,),
            candidate_replacement_beliefs=(r1, r2),
        )


def test_invalid_conditions_key_rejected() -> None:
    import pytest
    ev = _make_evidence()
    b1 = _make_belief("b1", "ev1")
    c1 = _make_condition("c1")
    with pytest.raises(ValueError, match="not a candidate belief id"):
        SharedCandidateView(
            instance_id="x", query_id="q", query="q",
            evidence_context=(ev,),
            candidate_beliefs=(b1,),
            candidate_replacement_beliefs=(),
            candidate_conditions_by_belief={"nonexistent": (c1,)},
        )


def test_duplicate_condition_ids_in_belief_rejected() -> None:
    import pytest
    ev = _make_evidence()
    b1 = _make_belief("b1", "ev1")
    c1 = _make_condition("c_dup")
    c2 = _make_condition("c_dup")
    with pytest.raises(ValueError, match="Duplicate condition_ids"):
        SharedCandidateView(
            instance_id="x", query_id="q", query="q",
            evidence_context=(ev,),
            candidate_beliefs=(b1,),
            candidate_replacement_beliefs=(),
            candidate_conditions_by_belief={"b1": (c1, c2)},
        )


def test_contracts_do_not_import_legacy_types() -> None:
    import retracemem.methods.contracts as mod

    source = open(mod.__file__).read()
    assert "RelationPrediction" not in source
    assert "RelationType" not in source
    assert "EpisodicEvidence" not in source
    assert "Belief," not in source  # not the legacy Belief type

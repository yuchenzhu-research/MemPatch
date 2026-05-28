from __future__ import annotations

import json

import pytest

from retracemem.schemas import BeliefNode, EvidenceNode
from retracemem.verifier.typed_edge_response_parser import EdgeTargetSpace, parse_typed_edge_response


def _evidence() -> EvidenceNode:
    return EvidenceNode(
        evidence_id="ev_new",
        session_id="s1",
        timestamp="2026-01-01T00:00:00Z",
        text="User changed plans.",
        source_dataset="test",
        source_pointer="ptr",
    )


def _replacement(source_id: str = "ev_new") -> BeliefNode:
    return BeliefNode(
        belief_id="b_rep",
        proposition="Replacement belief.",
        source_evidence_ids=(source_id,),
    )


def _target_space(single: bool = False, grounded: bool = True) -> EdgeTargetSpace:
    return EdgeTargetSpace(
        valid_belief_ids=frozenset({"b1", "b2"}),
        valid_condition_ids=frozenset({"c1"}),
        replacement_map={"b_rep": _replacement("ev_new" if grounded else "ev_old")},
        single_belief_id="b1" if single else None,
    )


def test_empty_edges_parse() -> None:
    edges = parse_typed_edge_response(
        json.dumps({"edges": []}),
        new_evidence=_evidence(),
        target_space=_target_space(),
        call_id="call_1",
        prompt_version="test_v1",
    )
    assert edges == []


@pytest.mark.parametrize("single", [True, False])
def test_invalid_condition_target_rejected_in_both_modes(single: bool) -> None:
    response = json.dumps({"edges": [
        {"edge_type": "BLOCKS", "target_id": "c_bad", "rationale": "x"}
    ]})
    with pytest.raises(ValueError, match="unknown condition"):
        parse_typed_edge_response(
            response,
            new_evidence=_evidence(),
            target_space=_target_space(single=single),
            call_id="call_1",
            prompt_version="test_v1",
        )


@pytest.mark.parametrize("single", [True, False])
def test_invalid_belief_target_rejected_in_both_modes(single: bool) -> None:
    response = json.dumps({"edges": [
        {"edge_type": "REAFFIRMS", "target_id": "b_bad", "rationale": "x"}
    ]})
    with pytest.raises(ValueError, match="candidate belief"):
        parse_typed_edge_response(
            response,
            new_evidence=_evidence(),
            target_space=_target_space(single=single),
            call_id="call_1",
            prompt_version="test_v1",
        )


def test_single_belief_mode_rejects_other_valid_belief() -> None:
    response = json.dumps({"edges": [
        {"edge_type": "REAFFIRMS", "target_id": "b2", "rationale": "x"}
    ]})
    with pytest.raises(ValueError, match="must target candidate belief"):
        parse_typed_edge_response(
            response,
            new_evidence=_evidence(),
            target_space=_target_space(single=True),
            call_id="call_1",
            prompt_version="test_v1",
        )


@pytest.mark.parametrize("single", [True, False])
def test_ungrounded_supersedes_rejected_in_both_modes(single: bool) -> None:
    response = json.dumps({"edges": [
        {"edge_type": "SUPERSEDES", "target_id": "b1", "replacement_belief_id": "b_rep", "rationale": "x"}
    ]})
    with pytest.raises(ValueError, match="not grounded"):
        parse_typed_edge_response(
            response,
            new_evidence=_evidence(),
            target_space=_target_space(single=single, grounded=False),
            call_id="call_1",
            prompt_version="test_v1",
        )


@pytest.mark.parametrize("single", [True, False])
def test_duplicate_deterministic_edge_rejected_in_both_modes(single: bool) -> None:
    response = json.dumps({"edges": [
        {"edge_type": "REAFFIRMS", "target_id": "b1", "rationale": "x"},
        {"edge_type": "REAFFIRMS", "target_id": "b1", "rationale": "y"},
    ]})
    with pytest.raises(ValueError, match="Duplicate edge"):
        parse_typed_edge_response(
            response,
            new_evidence=_evidence(),
            target_space=_target_space(single=single),
            call_id="call_1",
            prompt_version="test_v1",
        )

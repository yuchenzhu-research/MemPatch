"""Contract tests for PromptEvidenceEdgeVerifier using MockLLMProvider."""
from __future__ import annotations

import json
import os

import pytest

from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.providers.base import MockLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.schemas import BeliefNode, ConditionNode, EvidenceNode
from retracemem.verifier.prompt_evidence_edge_verifier import PromptEvidenceEdgeVerifier


def _make_evidence(eid: str = "ev_leg") -> EvidenceNode:
    return EvidenceNode(
        evidence_id=eid,
        session_id="s1",
        timestamp="2026-01-02T00:00:00Z",
        text="The user broke their leg.",
        source_dataset="test",
        source_pointer="ptr",
    )


def _make_belief(bid: str = "b_bike") -> BeliefNode:
    return BeliefNode(
        belief_id=bid,
        proposition="The user commutes by bicycle.",
        source_evidence_ids=("ev1",),
    )


def _make_replacement(bid: str = "b_car", grounded_evidence: str = "ev_leg") -> BeliefNode:
    return BeliefNode(
        belief_id=bid,
        proposition="The user commutes by car.",
        source_evidence_ids=(grounded_evidence,),
    )


def _make_condition(cid: str = "c_leg") -> ConditionNode:
    return ConditionNode(condition_id=cid, scope_id="user1", text="User is physically able.")


def _make_verifier(
    response: str,
    tmp_path: str,
    status: str = "success",
) -> PromptEvidenceEdgeVerifier:
    mock = MockLLMProvider(default_response=response, status=status)
    cache = JSONLCache(os.path.join(tmp_path, "cache.jsonl"))
    client = CachedLLMClient(cache=cache, provider_client=mock)
    return PromptEvidenceEdgeVerifier(client=client, model_id="mock", provider="mock")


def test_valid_blocks_parse(tmp_path: str) -> None:
    response = json.dumps({
        "edges": [
            {
                "edge_type": "BLOCKS",
                "target_id": "c_leg",
                "replacement_belief_id": None,
                "rationale": "Broken leg blocks cycling ability.",
                "confidence": 0.9,
            }
        ]
    })
    verifier = _make_verifier(response, str(tmp_path))
    edges = verifier.verify_edges(
        new_evidence=_make_evidence(),
        candidate_belief=_make_belief(),
        candidate_replacement_beliefs=(_make_replacement(),),
        candidate_conditions=(_make_condition(),),
        temporal_context=(),
    )

    assert len(edges) == 1
    assert edges[0].edge_type.value == "BLOCKS"
    assert edges[0].target_kind == "condition"
    assert edges[0].target_id == "c_leg"
    assert edges[0].model_call_trace_id is not None
    assert edges[0].edge_id.startswith("ee-")


def test_valid_supersedes_parse(tmp_path: str) -> None:
    response = json.dumps({
        "edges": [
            {
                "edge_type": "SUPERSEDES",
                "target_id": "b_bike",
                "replacement_belief_id": "b_car",
                "rationale": "User switched to car.",
                "confidence": 0.95,
            }
        ]
    })
    verifier = _make_verifier(response, str(tmp_path))
    edges = verifier.verify_edges(
        new_evidence=_make_evidence(),
        candidate_belief=_make_belief(),
        candidate_replacement_beliefs=(_make_replacement(),),
        candidate_conditions=(_make_condition(),),
        temporal_context=(),
    )

    assert len(edges) == 1
    assert edges[0].edge_type.value == "SUPERSEDES"
    assert edges[0].replacement_belief_id == "b_car"


def test_deterministic_edge_id_under_replay(tmp_path: str) -> None:
    response = json.dumps({
        "edges": [
            {"edge_type": "BLOCKS", "target_id": "c_leg", "rationale": "r", "confidence": 0.8}
        ]
    })
    v1 = _make_verifier(response, str(tmp_path / "a"))
    v2 = _make_verifier(response, str(tmp_path / "b"))
    e1 = v1.verify_edges(
        new_evidence=_make_evidence(),
        candidate_belief=_make_belief(),
        candidate_replacement_beliefs=(_make_replacement(),),
        candidate_conditions=(_make_condition(),),
        temporal_context=(),
    )
    e2 = v2.verify_edges(
        new_evidence=_make_evidence(),
        candidate_belief=_make_belief(),
        candidate_replacement_beliefs=(_make_replacement(),),
        candidate_conditions=(_make_condition(),),
        temporal_context=(),
    )
    assert e1[0].edge_id == e2[0].edge_id


def test_supersedes_ungrounded_replacement_rejected(tmp_path: str) -> None:
    response = json.dumps({
        "edges": [
            {
                "edge_type": "SUPERSEDES",
                "target_id": "b_bike",
                "replacement_belief_id": "b_car",
                "rationale": "Switch.",
            }
        ]
    })
    ungrounded_replacement = BeliefNode(
        belief_id="b_car",
        proposition="The user commutes by car.",
        source_evidence_ids=("ev_old",),
    )
    verifier = _make_verifier(response, str(tmp_path))
    with pytest.raises(ValueError, match="not grounded in current evidence"):
        verifier.verify_edges(
            new_evidence=_make_evidence(),
            candidate_belief=_make_belief(),
            candidate_replacement_beliefs=(ungrounded_replacement,),
            candidate_conditions=(_make_condition(),),
            temporal_context=(),
        )


def test_fabricated_replacement_rejected(tmp_path: str) -> None:
    response = json.dumps({
        "edges": [
            {
                "edge_type": "SUPERSEDES",
                "target_id": "b_bike",
                "replacement_belief_id": "b_nonexistent",
                "rationale": "Fabricated.",
                "confidence": 0.5,
            }
        ]
    })
    verifier = _make_verifier(response, str(tmp_path))
    with pytest.raises(ValueError, match="unknown replacement belief"):
        verifier.verify_edges(
            new_evidence=_make_evidence(),
            candidate_belief=_make_belief(),
            candidate_replacement_beliefs=(_make_replacement(),),
            candidate_conditions=(_make_condition(),),
            temporal_context=(),
        )


def test_supersedes_missing_replacement_rejected(tmp_path: str) -> None:
    response = json.dumps({
        "edges": [
            {
                "edge_type": "SUPERSEDES",
                "target_id": "b_bike",
                "replacement_belief_id": None,
                "rationale": "Missing.",
            }
        ]
    })
    verifier = _make_verifier(response, str(tmp_path))
    with pytest.raises(ValueError, match="missing replacement_belief_id"):
        verifier.verify_edges(
            new_evidence=_make_evidence(),
            candidate_belief=_make_belief(),
            candidate_replacement_beliefs=(_make_replacement(),),
            candidate_conditions=(_make_condition(),),
            temporal_context=(),
        )


def test_unsupported_condition_target_rejected(tmp_path: str) -> None:
    response = json.dumps({
        "edges": [
            {"edge_type": "BLOCKS", "target_id": "c_nonexistent", "rationale": "Bad."}
        ]
    })
    verifier = _make_verifier(response, str(tmp_path))
    with pytest.raises(ValueError, match="unknown condition"):
        verifier.verify_edges(
            new_evidence=_make_evidence(),
            candidate_belief=_make_belief(),
            candidate_replacement_beliefs=(),
            candidate_conditions=(_make_condition(),),
            temporal_context=(),
        )


def test_reaffirms_wrong_target_rejected(tmp_path: str) -> None:
    response = json.dumps({
        "edges": [
            {"edge_type": "REAFFIRMS", "target_id": "b_wrong", "rationale": "Bad."}
        ]
    })
    verifier = _make_verifier(response, str(tmp_path))
    with pytest.raises(ValueError, match="must target candidate belief"):
        verifier.verify_edges(
            new_evidence=_make_evidence(),
            candidate_belief=_make_belief(),
            candidate_replacement_beliefs=(),
            candidate_conditions=(),
            temporal_context=(),
        )


def test_duplicate_edge_rejected(tmp_path: str) -> None:
    response = json.dumps({
        "edges": [
            {"edge_type": "BLOCKS", "target_id": "c_leg", "rationale": "r1"},
            {"edge_type": "BLOCKS", "target_id": "c_leg", "rationale": "r2"},
        ]
    })
    verifier = _make_verifier(response, str(tmp_path))
    with pytest.raises(ValueError, match="Duplicate edge"):
        verifier.verify_edges(
            new_evidence=_make_evidence(),
            candidate_belief=_make_belief(),
            candidate_replacement_beliefs=(),
            candidate_conditions=(_make_condition(),),
            temporal_context=(),
        )


def test_invalid_confidence_rejected(tmp_path: str) -> None:
    response = json.dumps({
        "edges": [
            {"edge_type": "UNCERTAIN", "target_id": "b_bike", "rationale": "r", "confidence": 2.0}
        ]
    })
    verifier = _make_verifier(response, str(tmp_path))
    with pytest.raises(ValueError, match="Confidence must be"):
        verifier.verify_edges(
            new_evidence=_make_evidence(),
            candidate_belief=_make_belief(),
            candidate_replacement_beliefs=(),
            candidate_conditions=(),
            temporal_context=(),
        )


def test_malformed_json_failure(tmp_path: str) -> None:
    verifier = _make_verifier("not json", str(tmp_path))
    with pytest.raises((json.JSONDecodeError, ValueError)):
        verifier.verify_edges(
            new_evidence=_make_evidence(),
            candidate_belief=_make_belief(),
            candidate_replacement_beliefs=(),
            candidate_conditions=(),
            temporal_context=(),
        )


def test_api_failure_raises(tmp_path: str) -> None:
    verifier = _make_verifier("", str(tmp_path), status="failure")
    with pytest.raises(ValueError, match="Evidence-edge prediction failed"):
        verifier.verify_edges(
            new_evidence=_make_evidence(),
            candidate_belief=_make_belief(),
            candidate_replacement_beliefs=(),
            candidate_conditions=(),
            temporal_context=(),
        )


def test_trace_metadata_present(tmp_path: str) -> None:
    response = json.dumps({
        "edges": [
            {"edge_type": "UNCERTAIN", "target_id": "b_bike", "rationale": "Unclear.", "confidence": 0.3}
        ]
    })
    verifier = _make_verifier(response, str(tmp_path))
    edges = verifier.verify_edges(
        new_evidence=_make_evidence(),
        candidate_belief=_make_belief(),
        candidate_replacement_beliefs=(),
        candidate_conditions=(),
        temporal_context=(),
    )
    assert edges[0].model_call_trace_id is not None
    assert edges[0].verifier == "evidence_edge_prediction_v0"

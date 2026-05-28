"""Contract tests for PromptRequirementInducer using MockLLMProvider."""
from __future__ import annotations

import json
import os

import pytest

from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.providers.base import MockLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.schemas import BeliefNode, EvidenceNode
from retracemem.verifier.prompt_requirement_inducer import PromptRequirementInducer


def _make_evidence(eid: str = "ev1") -> EvidenceNode:
    return EvidenceNode(
        evidence_id=eid,
        session_id="s1",
        timestamp="2026-01-01T00:00:00Z",
        text="The user commutes by bicycle.",
        source_dataset="test",
        source_pointer="ptr",
    )


def _make_belief(bid: str = "b_bike", scope_id: str = "user1") -> BeliefNode:
    return BeliefNode(
        belief_id=bid,
        proposition="The user commutes by bicycle.",
        source_evidence_ids=("ev1",),
        metadata={"scope_id": scope_id},
    )


def _make_inducer(
    response: str,
    tmp_path: str,
    status: str = "success",
) -> PromptRequirementInducer:
    mock = MockLLMProvider(default_response=response, status=status)
    cache = JSONLCache(os.path.join(tmp_path, "cache.jsonl"))
    client = CachedLLMClient(cache=cache, provider_client=mock)
    return PromptRequirementInducer(client=client, model_id="mock", provider="mock")


def test_valid_parse(tmp_path: str) -> None:
    response = json.dumps({
        "requirements": [
            {
                "condition_text": "User is physically able to cycle.",
                "rationale": "Cycling requires physical ability.",
                "confidence": 0.85,
            }
        ]
    })
    inducer = _make_inducer(response, str(tmp_path))
    proposals = inducer.induce_requirements(_make_belief(), (_make_evidence(),))

    assert len(proposals) == 1
    cond = proposals[0].condition
    dep = proposals[0].dependency_edge

    assert cond.condition_id.startswith("c-")
    assert cond.scope_id == "user1"
    assert dep.belief_id == "b_bike"
    assert dep.condition_id == cond.condition_id
    assert dep.edge_type == "REQUIRES"
    assert dep.inducer == "requirement_induction_v0"
    assert dep.model_call_trace_id is not None
    assert "ev1" in dep.supporting_evidence_ids


def test_deterministic_id_under_replay(tmp_path: str) -> None:
    response = json.dumps({
        "requirements": [
            {"condition_text": "User is physically able.", "rationale": "r", "confidence": 0.5}
        ]
    })
    inducer1 = _make_inducer(response, str(tmp_path / "a"))
    inducer2 = _make_inducer(response, str(tmp_path / "b"))
    p1 = inducer1.induce_requirements(_make_belief(), (_make_evidence(),))
    p2 = inducer2.induce_requirements(_make_belief(), (_make_evidence(),))
    assert p1[0].condition.condition_id == p2[0].condition.condition_id
    assert p1[0].dependency_edge.edge_id == p2[0].dependency_edge.edge_id


def test_missing_scope_in_belief_fails(tmp_path: str) -> None:
    response = json.dumps({"requirements": []})
    inducer = _make_inducer(response, str(tmp_path))
    belief_no_scope = BeliefNode(
        belief_id="b1",
        proposition="Test.",
        source_evidence_ids=("ev1",),
    )
    with pytest.raises(ValueError, match="explicit scope_id"):
        inducer.induce_requirements(belief_no_scope, (_make_evidence(),))


def test_default_scope_not_used(tmp_path: str) -> None:
    response = json.dumps({"requirements": []})
    inducer = _make_inducer(response, str(tmp_path))
    belief_default = BeliefNode(
        belief_id="b1",
        proposition="Test.",
        source_evidence_ids=("ev1",),
        metadata={"scope_id": ""},
    )
    with pytest.raises(ValueError, match="explicit scope_id"):
        inducer.induce_requirements(belief_default, (_make_evidence(),))


def test_malformed_json_failure(tmp_path: str) -> None:
    inducer = _make_inducer("not json", str(tmp_path))
    with pytest.raises((json.JSONDecodeError, ValueError)):
        inducer.induce_requirements(_make_belief(), (_make_evidence(),))


def test_missing_requirements_key(tmp_path: str) -> None:
    inducer = _make_inducer(json.dumps({"items": []}), str(tmp_path))
    with pytest.raises(ValueError, match="missing 'requirements' key"):
        inducer.induce_requirements(_make_belief(), (_make_evidence(),))


def test_empty_condition_text_fails(tmp_path: str) -> None:
    response = json.dumps({"requirements": [{"condition_text": "  "}]})
    inducer = _make_inducer(response, str(tmp_path))
    with pytest.raises(ValueError, match="missing or empty condition_text"):
        inducer.induce_requirements(_make_belief(), (_make_evidence(),))


def test_duplicate_condition_fails(tmp_path: str) -> None:
    response = json.dumps({"requirements": [
        {"condition_text": "User can cycle.", "rationale": "r"},
        {"condition_text": "User can cycle.", "rationale": "r2"},
    ]})
    inducer = _make_inducer(response, str(tmp_path))
    with pytest.raises(ValueError, match="Duplicate normalized condition"):
        inducer.induce_requirements(_make_belief(), (_make_evidence(),))


def test_invalid_confidence_fails(tmp_path: str) -> None:
    response = json.dumps({"requirements": [
        {"condition_text": "Test.", "rationale": "r", "confidence": -0.1}
    ]})
    inducer = _make_inducer(response, str(tmp_path))
    with pytest.raises(ValueError, match="Confidence must be"):
        inducer.induce_requirements(_make_belief(), (_make_evidence(),))


def test_api_failure_raises(tmp_path: str) -> None:
    inducer = _make_inducer("", str(tmp_path), status="failure")
    with pytest.raises(ValueError, match="Requirement induction failed"):
        inducer.induce_requirements(_make_belief(), (_make_evidence(),))


def test_scope_from_belief_metadata(tmp_path: str) -> None:
    response = json.dumps({
        "requirements": [
            {"condition_text": "Test condition", "rationale": "r", "confidence": 0.5}
        ]
    })
    inducer = _make_inducer(response, str(tmp_path))
    proposals = inducer.induce_requirements(_make_belief(scope_id="custom_scope"), (_make_evidence(),))
    assert proposals[0].condition.scope_id == "custom_scope"

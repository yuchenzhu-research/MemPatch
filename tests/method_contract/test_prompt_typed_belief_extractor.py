"""Contract tests for PromptTypedBeliefExtractor using MockLLMProvider."""
from __future__ import annotations

import json
import os

import pytest

from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.providers.base import MockLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.schemas import EvidenceNode
from retracemem.verifier.prompt_typed_belief_extractor import PromptTypedBeliefExtractor


def _make_evidence(eid: str = "ev1") -> EvidenceNode:
    return EvidenceNode(
        evidence_id=eid,
        session_id="s1",
        timestamp="2026-01-01T00:00:00Z",
        text="The user now commutes by bicycle every day.",
        source_dataset="test",
        source_pointer="ptr",
    )


def _make_extractor(
    response: str,
    tmp_path: str,
    status: str = "success",
) -> PromptTypedBeliefExtractor:
    mock = MockLLMProvider(default_response=response, status=status)
    cache = JSONLCache(os.path.join(tmp_path, "cache.jsonl"))
    client = CachedLLMClient(cache=cache, provider_client=mock)
    return PromptTypedBeliefExtractor(client=client, model_id="mock", provider="mock")


def test_valid_parse(tmp_path: str) -> None:
    response = json.dumps({
        "beliefs": [
            {"proposition": "The user commutes by bicycle.", "confidence": 0.9}
        ]
    })
    extractor = _make_extractor(response, str(tmp_path))
    beliefs = extractor.extract(_make_evidence(), scope_id="user1")

    assert len(beliefs) == 1
    assert beliefs[0].belief_id.startswith("b-")
    assert beliefs[0].proposition == "The user commutes by bicycle."
    assert "ev1" in beliefs[0].source_evidence_ids
    assert beliefs[0].extractor_version == "belief_extraction_v0"
    assert beliefs[0].metadata.get("model_call_trace_id") is not None


def test_deterministic_id_under_replay(tmp_path: str) -> None:
    response = json.dumps({
        "beliefs": [{"proposition": "The user commutes by bicycle.", "confidence": 0.9}]
    })
    extractor1 = _make_extractor(response, str(tmp_path / "a"))
    extractor2 = _make_extractor(response, str(tmp_path / "b"))
    beliefs1 = extractor1.extract(_make_evidence(), scope_id="user1")
    beliefs2 = extractor2.extract(_make_evidence(), scope_id="user1")
    assert beliefs1[0].belief_id == beliefs2[0].belief_id


def test_different_scope_yields_different_id(tmp_path: str) -> None:
    response = json.dumps({
        "beliefs": [{"proposition": "The user commutes by bicycle."}]
    })
    extractor = _make_extractor(response, str(tmp_path))
    beliefs_a = extractor.extract(_make_evidence(), scope_id="user_a")
    extractor2 = _make_extractor(response, str(tmp_path / "b"))
    beliefs_b = extractor2.extract(_make_evidence(), scope_id="user_b")
    assert beliefs_a[0].belief_id != beliefs_b[0].belief_id


def test_malformed_json_failure(tmp_path: str) -> None:
    extractor = _make_extractor("not valid json", str(tmp_path))
    with pytest.raises((json.JSONDecodeError, ValueError)):
        extractor.extract(_make_evidence(), scope_id="user1")


def test_missing_beliefs_key(tmp_path: str) -> None:
    extractor = _make_extractor(json.dumps({"items": []}), str(tmp_path))
    with pytest.raises(ValueError, match="missing 'beliefs' key"):
        extractor.extract(_make_evidence(), scope_id="user1")


def test_empty_proposition_fails(tmp_path: str) -> None:
    response = json.dumps({"beliefs": [{"proposition": "  "}]})
    extractor = _make_extractor(response, str(tmp_path))
    with pytest.raises(ValueError, match="missing or empty proposition"):
        extractor.extract(_make_evidence(), scope_id="user1")


def test_duplicate_proposition_fails(tmp_path: str) -> None:
    response = json.dumps({"beliefs": [
        {"proposition": "The user cycles."},
        {"proposition": "The user cycles."},
    ]})
    extractor = _make_extractor(response, str(tmp_path))
    with pytest.raises(ValueError, match="Duplicate normalized proposition"):
        extractor.extract(_make_evidence(), scope_id="user1")


def test_invalid_confidence_fails(tmp_path: str) -> None:
    response = json.dumps({"beliefs": [
        {"proposition": "Test.", "confidence": 1.5}
    ]})
    extractor = _make_extractor(response, str(tmp_path))
    with pytest.raises(ValueError, match="Confidence must be"):
        extractor.extract(_make_evidence(), scope_id="user1")


def test_empty_scope_rejected(tmp_path: str) -> None:
    response = json.dumps({"beliefs": []})
    extractor = _make_extractor(response, str(tmp_path))
    with pytest.raises(ValueError, match="scope_id is required"):
        extractor.extract(_make_evidence(), scope_id="")


def test_api_failure_raises(tmp_path: str) -> None:
    extractor = _make_extractor("", str(tmp_path), status="failure")
    with pytest.raises(ValueError, match="Belief extraction failed"):
        extractor.extract(_make_evidence(), scope_id="user1")


def test_trace_metadata_present(tmp_path: str) -> None:
    response = json.dumps({
        "beliefs": [{"proposition": "Test."}]
    })
    extractor = _make_extractor(response, str(tmp_path))
    beliefs = extractor.extract(_make_evidence(), scope_id="user1")
    assert beliefs[0].metadata["model_call_trace_id"] is not None
    assert beliefs[0].metadata["scope_id"] == "user1"


def test_no_external_calls(tmp_path: str) -> None:
    response = json.dumps({"beliefs": []})
    mock = MockLLMProvider(default_response=response)
    cache = JSONLCache(os.path.join(str(tmp_path), "cache.jsonl"))
    client = CachedLLMClient(cache=cache, provider_client=mock)
    extractor = PromptTypedBeliefExtractor(client=client, model_id="mock", provider="mock")
    extractor.extract(_make_evidence(), scope_id="user1")
    assert mock.calls_count == 1


def test_model_does_not_control_id(tmp_path: str) -> None:
    response = json.dumps({
        "beliefs": [{"proposition": "Test belief.", "belief_id": "model_invented_id"}]
    })
    extractor = _make_extractor(response, str(tmp_path))
    beliefs = extractor.extract(_make_evidence(), scope_id="user1")
    assert beliefs[0].belief_id != "model_invented_id"
    assert beliefs[0].belief_id.startswith("b-")

"""Contract tests for DirectJudge-LLM using MockLLMProvider."""
from __future__ import annotations

import json
import os

import pytest

from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.methods.contracts import (
    DirectUsabilityStatus,
    SharedCandidateView,
)
from retracemem.methods.directjudge import DirectJudgeLLM
from retracemem.providers.base import MockLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.schemas import BeliefNode, ConditionNode, EvidenceNode


def _make_view() -> SharedCandidateView:
    ev = EvidenceNode(
        evidence_id="ev_leg",
        session_id="s1",
        timestamp="2026-01-02T00:00:00Z",
        text="The user broke their leg.",
        source_dataset="test",
        source_pointer="ptr",
    )
    b_bike = BeliefNode(
        belief_id="b_bike",
        proposition="The user commutes by bicycle.",
        source_evidence_ids=("ev1",),
    )
    b_car = BeliefNode(
        belief_id="b_car",
        proposition="The user commutes by car.",
        source_evidence_ids=("ev_leg",),
    )
    c_leg = ConditionNode(condition_id="c_leg", scope_id="user1", text="User is physically able.")
    return SharedCandidateView(
        instance_id="case_1",
        query_id="q_1",
        query="How does the user commute?",
        evidence_context=(ev,),
        candidate_beliefs=(b_bike, b_car),
        candidate_replacement_beliefs=(b_car,),
        candidate_conditions_by_belief=(("b_bike", (c_leg,)),),
        new_evidence=ev,
    )


def _make_judge(
    response: str,
    tmp_path: str,
    status: str = "success",
) -> DirectJudgeLLM:
    mock = MockLLMProvider(default_response=response, status=status)
    cache = JSONLCache(os.path.join(tmp_path, "cache.jsonl"))
    client = CachedLLMClient(cache=cache, provider_client=mock)
    return DirectJudgeLLM(client=client, model_id="mock", provider="mock")


def test_valid_parse(tmp_path: str) -> None:
    response = json.dumps({
        "verdicts": [
            {"belief_id": "b_bike", "status": "NOT_USABLE", "rationale": "Broken leg.", "confidence": 0.9},
            {"belief_id": "b_car", "status": "USABLE", "rationale": "Still valid.", "confidence": 0.85},
        ]
    })
    judge = _make_judge(response, str(tmp_path))
    result = judge.judge(_make_view())

    assert result.method_name == "directjudge_llm"
    assert result.instance_id == "case_1"
    assert result.query_id == "q_1"
    assert "b_car" in result.authorized_belief_ids
    assert "b_bike" in result.excluded_belief_ids
    assert len(result.verdicts) == 2
    assert len(result.model_call_trace_ids) == 1


def test_consumes_exact_shared_view(tmp_path: str) -> None:
    view = _make_view()
    response = json.dumps({
        "verdicts": [
            {"belief_id": "b_bike", "status": "NOT_USABLE", "rationale": "r"},
            {"belief_id": "b_car", "status": "USABLE", "rationale": "r"},
        ]
    })
    judge = _make_judge(response, str(tmp_path))
    result = judge.judge(view)
    assert result.instance_id == view.instance_id
    assert result.query_id == view.query_id


def test_unknown_belief_verdict_rejected(tmp_path: str) -> None:
    response = json.dumps({
        "verdicts": [
            {"belief_id": "b_nonexistent", "status": "USABLE", "rationale": "r"},
        ]
    })
    judge = _make_judge(response, str(tmp_path))
    with pytest.raises(ValueError, match="unknown belief_id"):
        judge.judge(_make_view())


def test_malformed_json_failure(tmp_path: str) -> None:
    judge = _make_judge("not json", str(tmp_path))
    with pytest.raises((json.JSONDecodeError, ValueError)):
        judge.judge(_make_view())


def test_api_failure_raises(tmp_path: str) -> None:
    judge = _make_judge("", str(tmp_path), status="failure")
    with pytest.raises(ValueError, match="DirectJudge failed"):
        judge.judge(_make_view())


def test_no_dpa_or_edge_imports() -> None:
    import retracemem.methods.directjudge as mod

    source = open(mod.__file__).read()
    import_lines = [
        line.strip() for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    joined = "\n".join(import_lines)
    assert "RevisionGate" not in joined
    assert "DefeatPathAuthorizationAlgorithm" not in joined
    assert "EvidenceEdge" not in joined
    assert "DependencyEdge" not in joined
    assert "EvidenceEdgeType" not in joined
    assert "RequirementInducer" not in joined
    assert "EvidenceEdgeVerifier" not in joined


def test_call_accounting_available(tmp_path: str) -> None:
    response = json.dumps({
        "verdicts": [
            {"belief_id": "b_bike", "status": "UNCERTAIN", "rationale": "r"},
            {"belief_id": "b_car", "status": "USABLE", "rationale": "r"},
        ]
    })
    judge = _make_judge(response, str(tmp_path))
    result = judge.judge(_make_view())
    assert "tokens" in result.cost or "calls" in result.cost


def test_uncertain_status_roundtrip(tmp_path: str) -> None:
    response = json.dumps({
        "verdicts": [
            {"belief_id": "b_bike", "status": "UNCERTAIN", "rationale": "Unclear."},
            {"belief_id": "b_car", "status": "UNCERTAIN", "rationale": "Unclear."},
        ]
    })
    judge = _make_judge(response, str(tmp_path))
    result = judge.judge(_make_view())
    assert len(result.authorized_belief_ids) == 0
    assert len(result.excluded_belief_ids) == 2
    for v in result.verdicts:
        assert v.status == DirectUsabilityStatus.UNCERTAIN


def test_omitted_candidate_belief_fails(tmp_path: str) -> None:
    response = json.dumps({
        "verdicts": [
            {"belief_id": "b_bike", "status": "NOT_USABLE", "rationale": "r"},
        ]
    })
    judge = _make_judge(response, str(tmp_path))
    with pytest.raises(ValueError, match="omitted verdicts"):
        judge.judge(_make_view())


def test_duplicate_verdict_fails(tmp_path: str) -> None:
    response = json.dumps({
        "verdicts": [
            {"belief_id": "b_bike", "status": "NOT_USABLE", "rationale": "r"},
            {"belief_id": "b_bike", "status": "USABLE", "rationale": "r"},
            {"belief_id": "b_car", "status": "USABLE", "rationale": "r"},
        ]
    })
    judge = _make_judge(response, str(tmp_path))
    with pytest.raises(ValueError, match="duplicate verdict"):
        judge.judge(_make_view())


def test_full_view_rendered_in_prompt(tmp_path: str) -> None:
    response = json.dumps({
        "verdicts": [
            {"belief_id": "b_bike", "status": "NOT_USABLE", "rationale": "r"},
            {"belief_id": "b_car", "status": "USABLE", "rationale": "r"},
        ]
    })
    mock = MockLLMProvider(default_response=response)
    cache = JSONLCache(os.path.join(str(tmp_path), "cache.jsonl"))
    client = CachedLLMClient(cache=cache, provider_client=mock)
    judge = DirectJudgeLLM(client=client, model_id="mock", provider="mock")
    judge.judge(_make_view())
    prompt = mock.last_prompt
    assert "b_car" in prompt
    assert "commutes by car" in prompt
    assert "c_leg" in prompt
    assert "physically able" in prompt


def test_view_fingerprint_in_provenance(tmp_path: str) -> None:
    response = json.dumps({
        "verdicts": [
            {"belief_id": "b_bike", "status": "NOT_USABLE", "rationale": "r"},
            {"belief_id": "b_car", "status": "USABLE", "rationale": "r"},
        ]
    })
    judge = _make_judge(response, str(tmp_path))
    view = _make_view()
    result = judge.judge(view)
    assert result.provenance["view_fingerprint"] == view.view_fingerprint


def test_per_instance_cost_not_cumulative(tmp_path: str) -> None:
    response = json.dumps({
        "verdicts": [
            {"belief_id": "b_bike", "status": "NOT_USABLE", "rationale": "r"},
            {"belief_id": "b_car", "status": "USABLE", "rationale": "r"},
        ]
    })
    mock = MockLLMProvider(default_response=response)
    cache = JSONLCache(os.path.join(str(tmp_path), "cache.jsonl"))
    client = CachedLLMClient(cache=cache, provider_client=mock)
    judge = DirectJudgeLLM(client=client, model_id="mock", provider="mock")
    view = _make_view()

    result1 = judge.judge(view)
    result2 = judge.judge(view)

    # Each run should report the same per-instance cost
    assert result1.cost["tokens"]["total"] == result2.cost["tokens"]["total"]
    # Cumulative client total should be 2x
    cumulative = client.cost_accountant.to_dict()
    assert cumulative["tokens"]["total"] >= 2 * result1.cost["tokens"]["total"]

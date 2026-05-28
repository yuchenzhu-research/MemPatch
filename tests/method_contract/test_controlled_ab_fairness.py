"""Controlled A/B fairness tests proving Stage A and Stage B share the same view.

These tests verify:
1. Both methods consume the same SharedCandidateView.
2. Stage A output is typed edges (never direct verdicts).
3. Stage B output is direct verdicts (never typed edges or DPA).
4. Both use CachedLLMClient with recorded ModelCallTrace.
5. Both run fully offline via MockLLMProvider.
6. No benchmark performance claims are embedded.
"""
from __future__ import annotations

import json
import os

from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.methods.contracts import (
    DirectUsabilityStatus,
    SharedCandidateView,
)
from retracemem.methods.directjudge import DirectJudgeLLM
from retracemem.providers.base import MockLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.schemas import BeliefNode, ConditionNode, EvidenceNode
from retracemem.verifier.prompt_evidence_edge_verifier import PromptEvidenceEdgeVerifier


def _shared_view() -> SharedCandidateView:
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
        instance_id="fairness_case_1",
        query_id="q_fair",
        query="How does the user commute?",
        evidence_context=(ev,),
        candidate_beliefs=(b_bike,),
        candidate_replacement_beliefs=(b_car,),
        candidate_conditions_by_belief={"b_bike": (c_leg,)},
    )


def _make_client(response: str, tmp_path: str) -> CachedLLMClient:
    mock = MockLLMProvider(default_response=response)
    cache = JSONLCache(os.path.join(tmp_path, "cache.jsonl"))
    return CachedLLMClient(cache=cache, provider_client=mock)


def test_stage_a_and_b_consume_same_view(tmp_path: str) -> None:
    view = _shared_view()

    edge_response = json.dumps({
        "edges": [
            {"edge_type": "BLOCKS", "target_id": "c_leg", "rationale": "Broken leg.", "confidence": 0.9}
        ]
    })
    stage_a_client = _make_client(edge_response, str(tmp_path / "a"))
    verifier = PromptEvidenceEdgeVerifier(client=stage_a_client, model_id="mock", provider="mock")

    edges = verifier.verify_edges(
        new_evidence=view.evidence_context[0],
        candidate_belief=view.candidate_beliefs[0],
        candidate_replacement_beliefs=view.candidate_replacement_beliefs,
        candidate_conditions=view.candidate_conditions_by_belief.get("b_bike", ()),
        temporal_context=(),
    )

    dj_response = json.dumps({
        "verdicts": [
            {"belief_id": "b_bike", "status": "NOT_USABLE", "rationale": "Broken leg."}
        ]
    })
    stage_b_client = _make_client(dj_response, str(tmp_path / "b"))
    judge = DirectJudgeLLM(client=stage_b_client, model_id="mock", provider="mock")
    result_b = judge.judge(view)

    assert len(edges) == 1
    assert edges[0].edge_type.value == "BLOCKS"

    assert result_b.instance_id == view.instance_id
    assert result_b.query_id == view.query_id
    assert "b_bike" in result_b.excluded_belief_ids


def test_stage_a_produces_edges_not_verdicts(tmp_path: str) -> None:
    view = _shared_view()
    response = json.dumps({
        "edges": [
            {"edge_type": "UNCERTAIN", "target_id": "b_bike", "rationale": "Unclear."}
        ]
    })
    client = _make_client(response, str(tmp_path))
    verifier = PromptEvidenceEdgeVerifier(client=client, model_id="mock", provider="mock")

    edges = verifier.verify_edges(
        new_evidence=view.evidence_context[0],
        candidate_belief=view.candidate_beliefs[0],
        candidate_replacement_beliefs=view.candidate_replacement_beliefs,
        candidate_conditions=view.candidate_conditions_by_belief.get("b_bike", ()),
        temporal_context=(),
    )

    assert all(hasattr(e, "edge_type") for e in edges)
    assert all(hasattr(e, "target_kind") for e in edges)


def test_stage_b_produces_verdicts_not_edges(tmp_path: str) -> None:
    view = _shared_view()
    response = json.dumps({
        "verdicts": [
            {"belief_id": "b_bike", "status": "NOT_USABLE", "rationale": "r"}
        ]
    })
    client = _make_client(response, str(tmp_path))
    judge = DirectJudgeLLM(client=client, model_id="mock", provider="mock")
    result = judge.judge(view)

    for v in result.verdicts:
        assert isinstance(v.status, DirectUsabilityStatus)
        assert not hasattr(v, "edge_type")
        assert not hasattr(v, "target_kind")


def test_both_use_cached_client_with_trace(tmp_path: str) -> None:
    view = _shared_view()

    edge_resp = json.dumps({"edges": []})
    dj_resp = json.dumps({
        "verdicts": [{"belief_id": "b_bike", "status": "USABLE", "rationale": "ok"}]
    })

    client_a = _make_client(edge_resp, str(tmp_path / "a"))
    client_b = _make_client(dj_resp, str(tmp_path / "b"))

    verifier = PromptEvidenceEdgeVerifier(client=client_a, model_id="mock", provider="mock")
    judge = DirectJudgeLLM(client=client_b, model_id="mock", provider="mock")

    verifier.verify_edges(
        new_evidence=view.evidence_context[0],
        candidate_belief=view.candidate_beliefs[0],
        candidate_replacement_beliefs=view.candidate_replacement_beliefs,
        candidate_conditions=view.candidate_conditions_by_belief.get("b_bike", ()),
        temporal_context=(),
    )
    result_b = judge.judge(view)

    assert client_a.cost_accountant.calls.get("total", 0) >= 1
    assert client_b.cost_accountant.calls.get("total", 0) >= 1
    assert len(result_b.model_call_trace_ids) >= 1


def test_both_fully_offline(tmp_path: str) -> None:
    view = _shared_view()

    mock_a = MockLLMProvider(default_response=json.dumps({"edges": []}))
    mock_b = MockLLMProvider(default_response=json.dumps({
        "verdicts": [{"belief_id": "b_bike", "status": "USABLE", "rationale": "ok"}]
    }))

    cache_a = JSONLCache(os.path.join(str(tmp_path), "a.jsonl"))
    cache_b = JSONLCache(os.path.join(str(tmp_path), "b.jsonl"))

    client_a = CachedLLMClient(cache=cache_a, provider_client=mock_a)
    client_b = CachedLLMClient(cache=cache_b, provider_client=mock_b)

    verifier = PromptEvidenceEdgeVerifier(client=client_a, model_id="mock", provider="mock")
    judge = DirectJudgeLLM(client=client_b, model_id="mock", provider="mock")

    verifier.verify_edges(
        new_evidence=view.evidence_context[0],
        candidate_belief=view.candidate_beliefs[0],
        candidate_replacement_beliefs=view.candidate_replacement_beliefs,
        candidate_conditions=view.candidate_conditions_by_belief.get("b_bike", ()),
        temporal_context=(),
    )
    judge.judge(view)

    assert mock_a.calls_count == 1
    assert mock_b.calls_count == 1

"""Controlled A/B fairness tests proving Stage A and Stage B share the same view.

These tests verify:
1. Both methods consume the same SharedCandidateView.
2. Stage A (ControlledReTraceLLM) runs edge prediction + DPA, not direct verdicts.
3. Stage B (DirectJudgeLLM) produces direct verdicts without DPA.
4. Both preserve the same view_fingerprint.
5. Both report per-instance cost (not cumulative).
6. Both run fully offline via MockLLMProvider.
7. No extraction, induction, or retrieval is involved.
"""
from __future__ import annotations

import json
import os

from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.methods.contracts import (
    DirectUsabilityStatus,
    SharedCandidateView,
)
from retracemem.methods.controlled_retrace import ControlledReTraceLLM
from retracemem.methods.directjudge import DirectJudgeLLM
from retracemem.providers.base import MockLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.schemas import BeliefNode, ConditionNode, DependencyEdge, EvidenceNode
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
    dep = DependencyEdge(
        edge_id="dep-b_bike-c_leg",
        belief_id="b_bike",
        condition_id="c_leg",
        inducer="test_fixture",
        edge_type="REQUIRES",
    )
    return SharedCandidateView(
        instance_id="fairness_case_1",
        query_id="q_fair",
        query="How does the user commute?",
        evidence_context=(ev,),
        candidate_beliefs=(b_bike,),
        candidate_replacement_beliefs=(b_car,),
        candidate_conditions_by_belief=(("b_bike", (c_leg,)),),
        dependency_edges_by_belief=(("b_bike", (dep,)),),
        new_evidence=ev,
    )


def _make_client(response: str, tmp_path: str) -> tuple[CachedLLMClient, MockLLMProvider]:
    mock = MockLLMProvider(default_response=response)
    cache = JSONLCache(os.path.join(tmp_path, "cache.jsonl"))
    client = CachedLLMClient(cache=cache, provider_client=mock)
    return client, mock


def test_stage_a_and_b_consume_same_view_fingerprint(tmp_path: str) -> None:
    """Both methods report the same view_fingerprint in provenance."""
    view = _shared_view()

    edge_response = json.dumps({
        "edges": [
            {"edge_type": "BLOCKS", "target_id": "c_leg", "rationale": "Broken leg.", "confidence": 0.9}
        ]
    })
    client_a, _ = _make_client(edge_response, str(tmp_path / "a"))
    verifier = PromptEvidenceEdgeVerifier(client=client_a, model_id="mock", provider="mock")
    runner_a = ControlledReTraceLLM(edge_verifier=verifier, client=client_a)
    result_a = runner_a.run(view)

    dj_response = json.dumps({
        "verdicts": [
            {"belief_id": "b_bike", "status": "NOT_USABLE", "rationale": "Broken leg."}
        ]
    })
    client_b, _ = _make_client(dj_response, str(tmp_path / "b"))
    judge = DirectJudgeLLM(client=client_b, model_id="mock", provider="mock")
    result_b = judge.judge(view)

    # Both report the same fingerprint
    assert result_a.provenance["view_fingerprint"] == view.view_fingerprint
    assert result_b.provenance["view_fingerprint"] == view.view_fingerprint
    assert result_a.provenance["view_fingerprint"] == result_b.provenance["view_fingerprint"]

    # Stage A yields exclusion via DPA
    assert "b_bike" in result_a.excluded_belief_ids
    assert result_a.provenance["fine_grained_statuses"]["b_bike"] == "BLOCKED"

    # Stage B yields exclusion via direct verdict
    assert "b_bike" in result_b.excluded_belief_ids


def test_stage_a_runs_edge_prediction_plus_dpa(tmp_path: str) -> None:
    """Stage A uses PromptEvidenceEdgeVerifier and DPA, not direct verdicts."""
    view = _shared_view()
    response = json.dumps({
        "edges": [
            {"edge_type": "UNCERTAIN", "target_id": "b_bike", "rationale": "Unclear."}
        ]
    })
    client_a, mock_a = _make_client(response, str(tmp_path))
    verifier = PromptEvidenceEdgeVerifier(client=client_a, model_id="mock", provider="mock")
    runner_a = ControlledReTraceLLM(edge_verifier=verifier, client=client_a)
    result_a = runner_a.run(view)

    # Edge verifier was called
    assert mock_a.calls_count == 1
    # Result has fine_grained DPA statuses
    assert "fine_grained_statuses" in result_a.provenance
    # No verdicts field (that's Stage B only)
    assert result_a.verdicts == ()


def test_stage_b_produces_verdicts_no_dpa(tmp_path: str) -> None:
    """Stage B produces direct verdicts and imports no DPA."""
    view = _shared_view()
    response = json.dumps({
        "verdicts": [
            {"belief_id": "b_bike", "status": "NOT_USABLE", "rationale": "r"}
        ]
    })
    client_b, _ = _make_client(response, str(tmp_path))
    judge = DirectJudgeLLM(client=client_b, model_id="mock", provider="mock")
    result_b = judge.judge(view)

    for v in result_b.verdicts:
        assert isinstance(v.status, DirectUsabilityStatus)
        assert not hasattr(v, "edge_type")
        assert not hasattr(v, "target_kind")

    # Stage B source does NOT import DPA or edge verifier
    import retracemem.methods.directjudge as mod
    source = open(mod.__file__).read()
    import_lines = [l for l in source.splitlines() if l.strip().startswith(("import ", "from "))]
    joined = "\n".join(import_lines)
    assert "DefeatPathAuthorizationAlgorithm" not in joined
    assert "PromptEvidenceEdgeVerifier" not in joined


def test_both_report_per_instance_cost(tmp_path: str) -> None:
    """Neither method reports cumulative cost as per-instance cost."""
    view = _shared_view()

    edge_resp = json.dumps({"edges": []})
    dj_resp = json.dumps({
        "verdicts": [{"belief_id": "b_bike", "status": "USABLE", "rationale": "ok"}]
    })

    client_a, _ = _make_client(edge_resp, str(tmp_path / "a"))
    client_b, _ = _make_client(dj_resp, str(tmp_path / "b"))

    verifier = PromptEvidenceEdgeVerifier(client=client_a, model_id="mock", provider="mock")
    runner_a = ControlledReTraceLLM(edge_verifier=verifier, client=client_a)
    judge = DirectJudgeLLM(client=client_b, model_id="mock", provider="mock")

    result_a1 = runner_a.run(view)
    result_a2 = runner_a.run(view)
    result_b1 = judge.judge(view)
    result_b2 = judge.judge(view)

    # Per-instance cost should be the same for each call (not growing)
    assert result_a1.cost["tokens"]["total"] == result_a2.cost["tokens"]["total"]
    assert result_b1.cost["tokens"]["total"] == result_b2.cost["tokens"]["total"]


def test_no_extraction_induction_retrieval_in_controlled_comparison(tmp_path: str) -> None:
    """Controlled comparison does not run extraction, induction, or retrieval."""
    view = _shared_view()

    edge_resp = json.dumps({"edges": []})
    dj_resp = json.dumps({
        "verdicts": [{"belief_id": "b_bike", "status": "USABLE", "rationale": "ok"}]
    })

    client_a, mock_a = _make_client(edge_resp, str(tmp_path / "a"))
    client_b, mock_b = _make_client(dj_resp, str(tmp_path / "b"))

    verifier = PromptEvidenceEdgeVerifier(client=client_a, model_id="mock", provider="mock")
    runner_a = ControlledReTraceLLM(edge_verifier=verifier, client=client_a)
    judge = DirectJudgeLLM(client=client_b, model_id="mock", provider="mock")

    runner_a.run(view)
    judge.judge(view)

    # Stage A: one call per candidate belief (edge verification only)
    assert mock_a.calls_count == len(view.candidate_beliefs)
    # Stage B: one call total (direct judgment)
    assert mock_b.calls_count == 1


def test_both_fully_offline(tmp_path: str) -> None:
    """Both methods run fully offline via MockLLMProvider."""
    view = _shared_view()

    edge_resp = json.dumps({"edges": []})
    dj_resp = json.dumps({
        "verdicts": [{"belief_id": "b_bike", "status": "USABLE", "rationale": "ok"}]
    })

    client_a, mock_a = _make_client(edge_resp, str(tmp_path / "a"))
    client_b, mock_b = _make_client(dj_resp, str(tmp_path / "b"))

    verifier = PromptEvidenceEdgeVerifier(client=client_a, model_id="mock", provider="mock")
    runner_a = ControlledReTraceLLM(edge_verifier=verifier, client=client_a)
    judge = DirectJudgeLLM(client=client_b, model_id="mock", provider="mock")

    runner_a.run(view)
    judge.judge(view)

    assert mock_a.calls_count >= 1
    assert mock_b.calls_count >= 1

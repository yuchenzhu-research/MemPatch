"""Contract tests for BatchedControlledReTraceLLM."""
from __future__ import annotations

import json
import os

import pytest

from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.methods.batched_controlled_retrace import BatchedControlledReTraceLLM
from retracemem.methods.contracts import SharedCandidateView
from retracemem.providers.base import MockLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.schemas import BeliefNode, ConditionNode, DependencyEdge, EvidenceNode
from retracemem.methods.controlled_retrace import ControlledReTraceLLM
from retracemem.verifier.prompt_evidence_edge_verifier import PromptEvidenceEdgeVerifier
from retracemem.verifier.prompt_batched_evidence_edge_verifier import PromptBatchedEvidenceEdgeVerifier


def _ev(eid: str = "ev_new", text: str = "User broke their leg.") -> EvidenceNode:
    return EvidenceNode(
        evidence_id=eid, session_id="s1",
        timestamp="2026-01-02T00:00:00Z", text=text,
        source_dataset="test", source_pointer="ptr",
    )


def _old_ev() -> EvidenceNode:
    return EvidenceNode(
        evidence_id="ev_old", session_id="s1",
        timestamp="2026-01-01T00:00:00Z", text="User commutes by bicycle.",
        source_dataset="test", source_pointer="ptr",
    )


def _belief(bid: str, prop: str = "The user commutes by bicycle.") -> BeliefNode:
    return BeliefNode(belief_id=bid, proposition=prop, source_evidence_ids=("ev_old",))


def _replacement(bid: str = "b_car") -> BeliefNode:
    return BeliefNode(belief_id=bid, proposition="The user commutes by car.", source_evidence_ids=("ev_new",))


def _cond(cid: str = "c_leg") -> ConditionNode:
    return ConditionNode(condition_id=cid, scope_id="user1", text="User is physically able.")


def _dep(bid: str, cid: str) -> DependencyEdge:
    return DependencyEdge(
        edge_id=f"dep-{bid}-{cid}", belief_id=bid,
        condition_id=cid, inducer="test_fixture", edge_type="REQUIRES",
    )


def _runner(response: str, tmp_path: str) -> tuple[BatchedControlledReTraceLLM, CachedLLMClient]:
    mock = MockLLMProvider(default_response=response)
    cache = JSONLCache(os.path.join(tmp_path, "cache.jsonl"))
    client = CachedLLMClient(cache=cache, provider_client=mock)
    verifier = PromptBatchedEvidenceEdgeVerifier(
        client=client, model_id="mock", provider="mock",
    )
    runner = BatchedControlledReTraceLLM(edge_verifier=verifier, client=client)
    return runner, client


def _multi_view(beliefs: list[BeliefNode], replacements=(), conditions_map=(), deps_map=()):
    ev_old = _old_ev()
    ev_new = _ev()
    return SharedCandidateView(
        instance_id="batch_case",
        query_id="q_batch",
        query="Test?",
        evidence_context=(ev_old, ev_new),
        new_evidence=ev_new,
        candidate_beliefs=tuple(beliefs),
        candidate_replacement_beliefs=tuple(replacements),
        candidate_conditions_by_belief=tuple(conditions_map),
        dependency_edges_by_belief=tuple(deps_map),
    )


# --- Core tests ---

def test_multiple_protected_beliefs_zero_edges(tmp_path):
    """Batch with multiple unrelated protected beliefs returns zero edges, all AUTHORIZED."""
    b1 = _belief("b_bike", "User cycles to work.")
    b2 = _belief("b_piano", "User plays jazz piano.")
    b3 = _belief("b_reads", "User reads novels.")
    view = _multi_view([b1, b2, b3])
    runner, _ = _runner(json.dumps({"edges": []}), str(tmp_path))
    result = runner.run(view)
    assert set(result.authorized_belief_ids) == {"b_bike", "b_piano", "b_reads"}
    assert len(result.excluded_belief_ids) == 0
    for bid in ("b_bike", "b_piano", "b_reads"):
        assert result.provenance["fine_grained_statuses"][bid] == "AUTHORIZED"


def test_one_affected_several_protected(tmp_path):
    """One affected belief blocked, several protected beliefs remain AUTHORIZED."""
    b_bike = _belief("b_bike", "User cycles to work.")
    b_piano = _belief("b_piano", "User plays jazz piano.")
    b_reads = _belief("b_reads", "User reads novels.")
    c_leg = _cond("c_leg")
    dep = _dep("b_bike", "c_leg")
    view = _multi_view(
        [b_bike, b_piano, b_reads],
        conditions_map=[("b_bike", (c_leg,))],
        deps_map=[("b_bike", (dep,))],
    )
    response = json.dumps({"edges": [
        {"edge_type": "BLOCKS", "target_id": "c_leg", "rationale": "Broken leg.", "confidence": 0.9}
    ]})
    runner, _ = _runner(response, str(tmp_path))
    result = runner.run(view)
    assert "b_bike" in result.excluded_belief_ids
    assert result.provenance["fine_grained_statuses"]["b_bike"] == "BLOCKED"
    assert "b_piano" in result.authorized_belief_ids
    assert "b_reads" in result.authorized_belief_ids


def test_uncertain_behaviour(tmp_path):
    """Genuine ambiguous belief preserves UNCERTAIN/UNRESOLVED."""
    b1 = _belief("b_move", "User will stay in SF next spring.")
    view = _multi_view([b1])
    response = json.dumps({"edges": [
        {"edge_type": "UNCERTAIN", "target_id": "b_move",
         "rationale": "Considering move but undecided.", "confidence": 0.5}
    ]})
    runner, _ = _runner(response, str(tmp_path))
    result = runner.run(view)
    assert "b_move" in result.excluded_belief_ids
    assert result.provenance["fine_grained_statuses"]["b_move"] == "UNRESOLVED"


def test_invalid_target_rejected(tmp_path):
    """Edge targeting unknown belief id is rejected by parser."""
    b1 = _belief("b_bike")
    view = _multi_view([b1])
    response = json.dumps({"edges": [
        {"edge_type": "REAFFIRMS", "target_id": "b_nonexistent", "rationale": "x", "confidence": 0.9}
    ]})
    runner, _ = _runner(response, str(tmp_path))
    with pytest.raises(ValueError, match="must target a candidate belief"):
        runner.run(view)


def test_duplicate_edges_rejected(tmp_path):
    """Duplicate edges for same target are rejected."""
    b1 = _belief("b_bike")
    view = _multi_view([b1])
    response = json.dumps({"edges": [
        {"edge_type": "REAFFIRMS", "target_id": "b_bike", "rationale": "a", "confidence": 0.9},
        {"edge_type": "REAFFIRMS", "target_id": "b_bike", "rationale": "b", "confidence": 0.8},
    ]})
    runner, _ = _runner(response, str(tmp_path))
    with pytest.raises(ValueError, match="Duplicate edge"):
        runner.run(view)


def test_supersedes_grounding_enforced(tmp_path):
    """SUPERSEDES without grounded replacement is rejected."""
    b1 = _belief("b_bike")
    view = _multi_view([b1])
    response = json.dumps({"edges": [
        {"edge_type": "SUPERSEDES", "target_id": "b_bike",
         "replacement_belief_id": "b_nonexistent", "rationale": "x", "confidence": 0.9}
    ]})
    runner, _ = _runner(response, str(tmp_path))
    with pytest.raises(ValueError, match="unknown replacement belief"):
        runner.run(view)


def test_provenance_complete(tmp_path):
    """Provenance contains execution_mode, batch info, and admitted anchors."""
    b1 = _belief("b_bike")
    c1 = _cond("c_leg")
    dep = _dep("b_bike", "c_leg")
    view = _multi_view([b1], conditions_map=[("b_bike", (c1,))], deps_map=[("b_bike", (dep,))])
    response = json.dumps({"edges": [
        {"edge_type": "BLOCKS", "target_id": "c_leg", "rationale": "r", "confidence": 0.9}
    ]})
    runner, _ = _runner(response, str(tmp_path))
    result = runner.run(view)
    prov = result.provenance
    assert prov["execution_mode"] == "batched_local_edges_v1"
    assert prov["batch_candidate_belief_count"] == 1
    assert prov["batch_model_call_trace_id"] != ""
    assert prov["prompt_version"] == "evidence_edge_prediction_batch_v1"
    assert len(prov["admitted_fixed_anchors"]) == 1
    assert len(prov["edge_proposals"]) == 1
    assert prov["edge_proposals"][0]["admitted"] is True


def test_dpa_matches_per_belief_for_same_edges(tmp_path):
    """For the same accepted edge set, batched and per-belief yield identical DPA results."""
    from retracemem.methods.controlled_retrace import ControlledReTraceLLM
    from retracemem.verifier.prompt_evidence_edge_verifier import PromptEvidenceEdgeVerifier

    b_bike = _belief("b_bike")
    c_leg = _cond("c_leg")
    dep = _dep("b_bike", "c_leg")
    ev_old = _old_ev()
    ev_new = _ev()
    view = SharedCandidateView(
        instance_id="eq_case", query_id="q_eq", query="How?",
        evidence_context=(ev_old, ev_new), new_evidence=ev_new,
        candidate_beliefs=(b_bike,),
        candidate_replacement_beliefs=(),
        candidate_conditions_by_belief=(("b_bike", (c_leg,)),),
        dependency_edges_by_belief=(("b_bike", (dep,)),),
    )
    blocks_resp = json.dumps({"edges": [
        {"edge_type": "BLOCKS", "target_id": "c_leg", "rationale": "r", "confidence": 0.9}
    ]})

    # Per-belief path
    mock_pb = MockLLMProvider(default_response=blocks_resp)
    cache_pb = JSONLCache(os.path.join(str(tmp_path), "cache_pb.jsonl"))
    client_pb = CachedLLMClient(cache=cache_pb, provider_client=mock_pb)
    verifier_pb = PromptEvidenceEdgeVerifier(client=client_pb, model_id="mock", provider="mock")
    runner_pb = ControlledReTraceLLM(edge_verifier=verifier_pb, client=client_pb)
    result_pb = runner_pb.run(view)

    # Batched path
    mock_ba = MockLLMProvider(default_response=blocks_resp)
    cache_ba = JSONLCache(os.path.join(str(tmp_path), "cache_ba.jsonl"))
    client_ba = CachedLLMClient(cache=cache_ba, provider_client=mock_ba)
    verifier_ba = PromptBatchedEvidenceEdgeVerifier(client=client_ba, model_id="mock", provider="mock")
    runner_ba = BatchedControlledReTraceLLM(edge_verifier=verifier_ba, client=client_ba)
    result_ba = runner_ba.run(view)

    assert set(result_pb.authorized_belief_ids) == set(result_ba.authorized_belief_ids)
    assert set(result_pb.excluded_belief_ids) == set(result_ba.excluded_belief_ids)
    assert result_pb.provenance["fine_grained_statuses"] == result_ba.provenance["fine_grained_statuses"]


def test_single_model_call_for_batch(tmp_path):
    """Batched path makes exactly one model call regardless of belief count."""
    b1 = _belief("b_bike")
    b2 = _belief("b_piano", "User plays piano.")
    b3 = _belief("b_reads", "User reads novels.")
    view = _multi_view([b1, b2, b3])
    mock = MockLLMProvider(default_response=json.dumps({"edges": []}))
    cache = JSONLCache(os.path.join(str(tmp_path), "cache.jsonl"))
    client = CachedLLMClient(cache=cache, provider_client=mock)
    verifier = PromptBatchedEvidenceEdgeVerifier(client=client, model_id="mock", provider="mock")
    runner = BatchedControlledReTraceLLM(edge_verifier=verifier, client=client)
    result = runner.run(view)
    assert mock.calls_count == 1
    assert len(result.model_call_trace_ids) == 1


def test_rejected_fixed_anchor_fails_loudly_in_both_wrappers(tmp_path):
    b1 = _belief("b_bike")
    c1 = _cond("c_leg")
    bad_dep = DependencyEdge(
        edge_id="dep_bad", belief_id="b_bike", condition_id="c_leg",
        inducer="", edge_type="REQUIRES",
    )
    ev_old = _old_ev()
    ev_new = _ev()
    view = SharedCandidateView(
        instance_id="bad_anchor", query_id="q_bad", query="How?",
        evidence_context=(ev_old, ev_new), new_evidence=ev_new,
        candidate_beliefs=(b1,),
        candidate_replacement_beliefs=(),
        candidate_conditions_by_belief=(("b_bike", (c1,)),),
        dependency_edges_by_belief=(("b_bike", (bad_dep,)),),
    )
    response = json.dumps({"edges": []})

    per_mock = MockLLMProvider(default_response=response)
    per_client = CachedLLMClient(JSONLCache(os.path.join(str(tmp_path), "p.jsonl")), per_mock)
    per_runner = ControlledReTraceLLM(
        edge_verifier=PromptEvidenceEdgeVerifier(per_client, model_id="mock", provider="mock"),
        client=per_client,
    )

    batched_mock = MockLLMProvider(default_response=response)
    batched_client = CachedLLMClient(JSONLCache(os.path.join(str(tmp_path), "b.jsonl")), batched_mock)
    batched_runner = BatchedControlledReTraceLLM(
        edge_verifier=PromptBatchedEvidenceEdgeVerifier(batched_client, model_id="mock", provider="mock"),
        client=batched_client,
    )

    with pytest.raises(ValueError, match="rejected by RevisionGate"):
        per_runner.run(view)
    with pytest.raises(ValueError, match="rejected by RevisionGate"):
        batched_runner.run(view)

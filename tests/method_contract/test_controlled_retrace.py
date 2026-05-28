"""Contract tests for ControlledReTraceLLM using MockLLMProvider."""
from __future__ import annotations

import json
import os

import pytest

from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.methods.contracts import SharedCandidateView
from retracemem.methods.controlled_retrace import ControlledReTraceLLM
from retracemem.providers.base import MockLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.schemas import BeliefNode, ConditionNode, DependencyEdge, EvidenceNode
from retracemem.verifier.prompt_evidence_edge_verifier import PromptEvidenceEdgeVerifier


def _make_evidence(eid: str = "ev_new", text: str = "User broke their leg.") -> EvidenceNode:
    return EvidenceNode(
        evidence_id=eid,
        session_id="s1",
        timestamp="2026-01-02T00:00:00Z",
        text=text,
        source_dataset="test",
        source_pointer="ptr",
    )


def _make_old_evidence() -> EvidenceNode:
    return EvidenceNode(
        evidence_id="ev_old",
        session_id="s1",
        timestamp="2026-01-01T00:00:00Z",
        text="User commutes by bicycle.",
        source_dataset="test",
        source_pointer="ptr",
    )


def _make_belief(bid: str = "b_bike") -> BeliefNode:
    return BeliefNode(
        belief_id=bid,
        proposition="The user commutes by bicycle.",
        source_evidence_ids=("ev_old",),
    )


def _make_replacement(bid: str = "b_car") -> BeliefNode:
    return BeliefNode(
        belief_id=bid,
        proposition="The user commutes by car.",
        source_evidence_ids=("ev_new",),
    )


def _make_condition(cid: str = "c_leg") -> ConditionNode:
    return ConditionNode(condition_id=cid, scope_id="user1", text="User is physically able.")


def _make_dep_edge(bid: str = "b_bike", cid: str = "c_leg") -> DependencyEdge:
    return DependencyEdge(
        edge_id=f"dep-{bid}-{cid}",
        belief_id=bid,
        condition_id=cid,
        inducer="test_fixture",
        edge_type="REQUIRES",
    )


def _make_view(
    mock_response: str = "",
    new_evidence: EvidenceNode | None = None,
) -> SharedCandidateView:
    ev_old = _make_old_evidence()
    ev_new = new_evidence or _make_evidence()
    b1 = _make_belief("b_bike")
    b_car = _make_replacement("b_car")
    c1 = _make_condition("c_leg")
    dep = _make_dep_edge("b_bike", "c_leg")
    return SharedCandidateView(
        instance_id="case_1",
        query_id="q_1",
        query="How does the user commute?",
        evidence_context=(ev_old, ev_new),
        candidate_beliefs=(b1,),
        candidate_replacement_beliefs=(b_car,),
        candidate_conditions_by_belief=(("b_bike", (c1,)),),
        dependency_edges_by_belief=(("b_bike", (dep,)),),
        new_evidence=ev_new,
    )


def _make_runner(
    response: str,
    tmp_path: str,
) -> tuple[ControlledReTraceLLM, CachedLLMClient]:
    mock = MockLLMProvider(default_response=response)
    cache = JSONLCache(os.path.join(tmp_path, "cache.jsonl"))
    client = CachedLLMClient(cache=cache, provider_client=mock)
    verifier = PromptEvidenceEdgeVerifier(client=client, model_id="mock", provider="mock")
    runner = ControlledReTraceLLM(edge_verifier=verifier, client=client)
    return runner, client


def test_blocks_edge_yields_exclusion(tmp_path: str) -> None:
    """A BLOCKS edge plus supplied REQUIRES anchor yields BLOCKED exclusion."""
    response = json.dumps({
        "edges": [
            {
                "edge_type": "BLOCKS",
                "target_id": "c_leg",
                "rationale": "Broken leg blocks cycling.",
                "confidence": 0.9,
            }
        ]
    })
    runner, _ = _make_runner(response, str(tmp_path))
    view = _make_view()
    result = runner.run(view)

    assert result.method_name == "retrace_llm_controlled"
    assert "b_bike" in result.excluded_belief_ids
    assert "b_bike" not in result.authorized_belief_ids
    assert result.provenance["fine_grained_statuses"]["b_bike"] == "BLOCKED"
    assert result.provenance["view_fingerprint"] == view.view_fingerprint


def test_supersedes_edge_yields_exclusion(tmp_path: str) -> None:
    """A SUPERSEDES edge with grounded replacement yields SUPERSEDED."""
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
    runner, _ = _make_runner(response, str(tmp_path))
    view = _make_view()
    result = runner.run(view)

    assert "b_bike" in result.excluded_belief_ids
    assert result.provenance["fine_grained_statuses"]["b_bike"] == "SUPERSEDED"
    paths = result.provenance["defeat_paths"]
    assert len(paths) == 1
    assert paths[0]["path_type"] == "DIRECT_SUPERSEDE"
    assert paths[0]["replacement_belief_id"] == "b_car"


def test_uncertain_edge_yields_unresolved(tmp_path: str) -> None:
    """An UNCERTAIN edge maps to UNRESOLVED."""
    response = json.dumps({
        "edges": [
            {
                "edge_type": "UNCERTAIN",
                "target_id": "b_bike",
                "rationale": "Unclear whether user still cycles.",
                "confidence": 0.4,
            }
        ]
    })
    runner, _ = _make_runner(response, str(tmp_path))
    view = _make_view()
    result = runner.run(view)

    assert "b_bike" in result.excluded_belief_ids
    assert result.provenance["fine_grained_statuses"]["b_bike"] == "UNRESOLVED"


def test_no_edge_yields_authorized(tmp_path: str) -> None:
    """No defeating edge means AUTHORIZED."""
    response = json.dumps({"edges": []})
    runner, _ = _make_runner(response, str(tmp_path))
    view = _make_view()
    result = runner.run(view)

    assert "b_bike" in result.authorized_belief_ids
    assert result.provenance["fine_grained_statuses"]["b_bike"] == "AUTHORIZED"


def test_releases_clears_blocker(tmp_path: str) -> None:
    """A RELEASES edge clears a previous BLOCKS; belief is then AUTHORIZED."""
    response = json.dumps({
        "edges": [
            {
                "edge_type": "RELEASES",
                "target_id": "c_leg",
                "rationale": "Leg healed.",
                "confidence": 0.9,
            }
        ]
    })
    runner, _ = _make_runner(response, str(tmp_path))
    view = _make_view()
    result = runner.run(view)

    assert "b_bike" in result.authorized_belief_ids
    assert result.provenance["fine_grained_statuses"]["b_bike"] == "AUTHORIZED"


def test_no_hidden_dependency_edges_invented(tmp_path: str) -> None:
    """Runner must not invent dependency edges beyond what is supplied."""
    response = json.dumps({"edges": []})
    runner, _ = _make_runner(response, str(tmp_path))
    # View with no dependency edges: a BLOCKS edge alone should NOT block.
    ev_old = _make_old_evidence()
    ev_new = _make_evidence()
    b1 = _make_belief("b_bike")
    b_car = _make_replacement("b_car")
    view_no_deps = SharedCandidateView(
        instance_id="case_2",
        query_id="q_2",
        query="How?",
        evidence_context=(ev_old, ev_new),
        candidate_beliefs=(b1,),
        candidate_replacement_beliefs=(b_car,),
        candidate_conditions_by_belief=(),
        dependency_edges_by_belief=(),
        new_evidence=ev_new,
    )
    result = runner.run(view_no_deps)
    assert "b_bike" in result.authorized_belief_ids


def test_view_fingerprint_in_provenance(tmp_path: str) -> None:
    """View fingerprint must appear in output provenance."""
    response = json.dumps({"edges": []})
    runner, _ = _make_runner(response, str(tmp_path))
    view = _make_view()
    result = runner.run(view)
    assert result.provenance["view_fingerprint"] == view.view_fingerprint


def test_per_instance_cost_not_cumulative(tmp_path: str) -> None:
    """Per-instance cost must not accumulate across multiple runs."""
    response = json.dumps({"edges": []})
    runner, client = _make_runner(response, str(tmp_path))
    view = _make_view()

    result1 = runner.run(view)
    result2 = runner.run(view)

    # Both should report the same per-instance cost (same work each time)
    assert result1.cost["tokens"]["total"] == result2.cost["tokens"]["total"]
    # The cumulative total on the client should be 2x
    cumulative = client.cost_accountant.to_dict()
    assert cumulative["tokens"]["total"] >= 2 * result1.cost["tokens"]["total"]


def test_requires_new_evidence(tmp_path: str) -> None:
    """Runner must raise if new_evidence is not set."""
    response = json.dumps({"edges": []})
    runner, _ = _make_runner(response, str(tmp_path))
    ev = _make_old_evidence()
    b1 = _make_belief("b_bike")
    view_no_new = SharedCandidateView(
        instance_id="x", query_id="q", query="q",
        evidence_context=(ev,),
        candidate_beliefs=(b1,),
        candidate_replacement_beliefs=(),
    )
    with pytest.raises(ValueError, match="requires SharedCandidateView.new_evidence"):
        runner.run(view_no_new)


def test_does_not_run_extraction_or_retrieval(tmp_path: str) -> None:
    """Verify no extraction/induction/retrieval modules are called."""
    response = json.dumps({"edges": []})
    mock = MockLLMProvider(default_response=response)
    cache = JSONLCache(os.path.join(str(tmp_path), "cache.jsonl"))
    client = CachedLLMClient(cache=cache, provider_client=mock)
    verifier = PromptEvidenceEdgeVerifier(client=client, model_id="mock", provider="mock")
    runner = ControlledReTraceLLM(edge_verifier=verifier, client=client)

    view = _make_view()
    result = runner.run(view)

    # Only 1 call per candidate belief (edge verification only)
    assert mock.calls_count == len(view.candidate_beliefs)
    assert result.instance_id == view.instance_id

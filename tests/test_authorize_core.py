from __future__ import annotations

from retracemem.authorization import EvidenceProposalBatch, authorize
from retracemem.methods.contracts import SharedCandidateView
from retracemem.schemas import BeliefNode, EvidenceEdge, EvidenceEdgeType, EvidenceNode


def test_authorize_supersedes_prior_belief() -> None:
    old_evidence = EvidenceNode(
        evidence_id="ev_old",
        session_id="s1",
        timestamp="2026-01-01T00:00:00Z",
        text="User works in Paris.",
        source_dataset="unit",
        source_pointer="old",
    )
    new_evidence = EvidenceNode(
        evidence_id="ev_new",
        session_id="s1",
        timestamp="2026-01-02T00:00:00Z",
        text="User moved to Berlin.",
        source_dataset="unit",
        source_pointer="new",
    )
    prior = BeliefNode(
        belief_id="b_old",
        proposition="User works in Paris.",
        source_evidence_ids=("ev_old",),
    )
    replacement = BeliefNode(
        belief_id="b_new",
        proposition="User works in Berlin.",
        source_evidence_ids=("ev_new",),
    )
    edge = EvidenceEdge(
        edge_id="edge_1",
        edge_type=EvidenceEdgeType.SUPERSEDES,
        evidence_id="ev_new",
        target_kind="belief",
        target_id="b_old",
        verifier="unit",
        replacement_belief_id="b_new",
    )
    view = SharedCandidateView(
        instance_id="i1",
        query_id="q1",
        query="Where does the user work?",
        new_evidence=new_evidence,
        evidence_context=(old_evidence, new_evidence),
        candidate_beliefs=(prior,),
        candidate_replacement_beliefs=(replacement,),
        candidate_conditions_by_belief=(),
        dependency_edges_by_belief=(),
    )

    result = authorize(view, (EvidenceProposalBatch(edges=(edge,)),))

    assert result.authorized_belief_ids == ()
    assert result.excluded_belief_ids == ("b_old",)
    assert result.trace["fine_grained_statuses"]["b_old"] == "SUPERSEDED"

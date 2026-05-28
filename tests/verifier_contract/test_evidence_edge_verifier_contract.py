from __future__ import annotations

import pytest
from retracemem.schemas import BeliefNode, EvidenceEdge, EvidenceEdgeType
from retracemem.verifier.evidence_edge_verifier import (
    HeuristicEvidenceEdgeVerifier,
    ManualEvidenceEdgeVerifier,
)
from tests.verifier_contract._helpers import belief, condition, evidence, evidence_edge


def test_manual_evidence_edge_verifier_returns_typed_edges() -> None:
    edge = evidence_edge(EvidenceEdgeType.REAFFIRMS, "belief", "belief_bike")
    verifier = ManualEvidenceEdgeVerifier((edge,))

    result = verifier.verify_edges(
        new_evidence=evidence("evidence_1", "The user still commutes by bicycle."),
        candidate_belief=belief("belief_bike", "The user commutes by bicycle."),
        candidate_replacement_beliefs=(),
        candidate_conditions=(),
        temporal_context=(),
    )

    assert isinstance(result[0], EvidenceEdge)
    assert result[0].edge_type == edge.edge_type
    assert result[0].target_kind == edge.target_kind
    assert result[0].target_id == edge.target_id


def test_heuristic_blocks_targets_condition_using_candidate_conditions() -> None:
    verifier = HeuristicEvidenceEdgeVerifier()
    condition_node = condition("condition:user_1:mobility", "current mobility ability")

    result = verifier.verify_edges(
        new_evidence=evidence("evidence_broken_leg", "The user broke their leg and is in a cast."),
        candidate_belief=belief("belief_bike", "The user commutes by bicycle."),
        candidate_replacement_beliefs=(),
        candidate_conditions=(condition_node,),
        temporal_context=(),
    )

    assert [edge.edge_type for edge in result] == [EvidenceEdgeType.BLOCKS]
    assert result[0].target_kind == "condition"
    assert result[0].target_id == condition_node.condition_id
    assert result[0].metadata["candidate_condition_id"] == condition_node.condition_id


def test_heuristic_releases_targets_condition_using_temporal_context() -> None:
    verifier = HeuristicEvidenceEdgeVerifier()
    condition_node = condition("condition:user_1:mobility", "current mobility ability")

    result = verifier.verify_edges(
        new_evidence=evidence("evidence_recovered", "The user recovered and is cleared to bike again."),
        candidate_belief=belief("belief_bike", "The user commutes by bicycle."),
        candidate_replacement_beliefs=(),
        candidate_conditions=(condition_node,),
        temporal_context=(
            evidence(
                "evidence_broken_leg",
                "The user broke their leg.",
                metadata={"edge_type": EvidenceEdgeType.BLOCKS.value},
            ),
        ),
    )

    assert [edge.edge_type for edge in result] == [EvidenceEdgeType.RELEASES]
    assert result[0].target_kind == "condition"
    assert result[0].target_id == condition_node.condition_id
    assert result[0].metadata["temporal_context_ids"] == ("evidence_broken_leg",)


def test_heuristic_release_requires_temporal_context() -> None:
    verifier = HeuristicEvidenceEdgeVerifier()

    result = verifier.verify_edges(
        new_evidence=evidence("evidence_recovered", "The user recovered and is cleared to bike again."),
        candidate_belief=belief("belief_bike", "The user commutes by bicycle."),
        candidate_replacement_beliefs=(),
        candidate_conditions=(condition("condition:user_1:mobility", "current mobility ability"),),
        temporal_context=(),
    )

    assert result == []


def test_heuristic_supersedes_targets_belief_and_includes_replacement() -> None:
    new_ev = evidence("evidence_address", "The user moved to 88 Cedar Avenue.")
    cand_belief = belief("belief_address", "The user lives at 14 Pine Street.")
    rep_belief = BeliefNode(
        belief_id="replacement_belief",
        proposition="The user lives at 88 Cedar Avenue.",
        source_evidence_ids=("evidence_address",),
    )
    result = HeuristicEvidenceEdgeVerifier().verify_edges(
        new_evidence=new_ev,
        candidate_belief=cand_belief,
        candidate_replacement_beliefs=(rep_belief,),
        candidate_conditions=(),
        temporal_context=(),
    )

    assert [edge.edge_type for edge in result] == [EvidenceEdgeType.SUPERSEDES]
    assert result[0].target_kind == "belief"
    assert result[0].target_id == "belief_address"
    assert result[0].replacement_belief_id == "replacement_belief"


def test_heuristic_supersedes_returns_no_edge_when_replacements_absent_or_not_grounded() -> None:
    new_ev = evidence("evidence_address", "The user moved to 88 Cedar Avenue.")
    cand_belief = belief("belief_address", "The user lives at 14 Pine Street.")
    
    # 1. Replacements absent
    result_absent = HeuristicEvidenceEdgeVerifier().verify_edges(
        new_evidence=new_ev,
        candidate_belief=cand_belief,
        candidate_replacement_beliefs=(),
        candidate_conditions=(),
        temporal_context=(),
    )
    assert result_absent == []

    # 2. Replacement not grounded (source_evidence_ids does not contain evidence_address)
    rep_not_grounded = BeliefNode(
        belief_id="replacement_belief",
        proposition="The user lives at 88 Cedar Avenue.",
        source_evidence_ids=("other_evidence",),
    )
    result_not_grounded = HeuristicEvidenceEdgeVerifier().verify_edges(
        new_evidence=new_ev,
        candidate_belief=cand_belief,
        candidate_replacement_beliefs=(rep_not_grounded,),
        candidate_conditions=(),
        temporal_context=(),
    )
    assert result_not_grounded == []

    # 3. Replacement same belief_id
    rep_same_id = BeliefNode(
        belief_id="belief_address",
        proposition="The user lives at 88 Cedar Avenue.",
        source_evidence_ids=("evidence_address",),
    )
    result_same_id = HeuristicEvidenceEdgeVerifier().verify_edges(
        new_evidence=new_ev,
        candidate_belief=cand_belief,
        candidate_replacement_beliefs=(rep_same_id,),
        candidate_conditions=(),
        temporal_context=(),
    )
    assert result_same_id == []

    # 4. Replacement no topic overlap
    rep_no_overlap = BeliefNode(
        belief_id="replacement_belief",
        proposition="The user likes apples.",
        source_evidence_ids=("evidence_address",),
    )
    result_no_overlap = HeuristicEvidenceEdgeVerifier().verify_edges(
        new_evidence=new_ev,
        candidate_belief=cand_belief,
        candidate_replacement_beliefs=(rep_no_overlap,),
        candidate_conditions=(),
        temporal_context=(),
    )
    assert result_no_overlap == []


def test_manual_supersedes_fails_loudly_when_not_grounded() -> None:
    edge = evidence_edge(EvidenceEdgeType.SUPERSEDES, "belief", "belief_address")
    # Note: evidence_edge helper creates a SUPERSEDES edge with replacement_belief_id="belief_replacement"
    # and evidence_id="evidence_1"
    
    verifier = ManualEvidenceEdgeVerifier((edge,))
    
    new_ev = evidence("evidence_1", "The user moved to 88 Cedar Avenue.")
    cand_belief = belief("belief_address", "The user lives at 14 Pine Street.")
    
    # Under valid grounded replacement candidate:
    valid_rep = BeliefNode(
        belief_id="belief_replacement",
        proposition="The user lives at 88 Cedar Avenue.",
        source_evidence_ids=("evidence_1",),
    )
    
    result = verifier.verify_edges(
        new_evidence=new_ev,
        candidate_belief=cand_belief,
        candidate_replacement_beliefs=(valid_rep,),
        candidate_conditions=(),
        temporal_context=(),
    )
    assert len(result) == 1
    assert result[0].replacement_belief_id == "belief_replacement"

    # Fails loudly (ValueError) if candidate replacements do not contain it
    with pytest.raises(ValueError) as excinfo:
        verifier.verify_edges(
            new_evidence=new_ev,
            candidate_belief=cand_belief,
            candidate_replacement_beliefs=(),
            candidate_conditions=(),
            temporal_context=(),
        )
    assert "either not in candidate_replacement_beliefs or not grounded" in str(excinfo.value)

    # Fails loudly if candidate replacement is present but not grounded in this evidence_id
    ungrounded_rep = BeliefNode(
        belief_id="belief_replacement",
        proposition="The user lives at 88 Cedar Avenue.",
        source_evidence_ids=("other_evidence",),
    )
    with pytest.raises(ValueError) as excinfo:
        verifier.verify_edges(
            new_evidence=new_ev,
            candidate_belief=cand_belief,
            candidate_replacement_beliefs=(ungrounded_rep,),
            candidate_conditions=(),
            temporal_context=(),
        )
    assert "either not in candidate_replacement_beliefs or not grounded" in str(excinfo.value)


def test_heuristic_reaffirms_targets_belief() -> None:
    result = HeuristicEvidenceEdgeVerifier().verify_edges(
        new_evidence=evidence("evidence_still_bikes", "The user still commutes by bicycle."),
        candidate_belief=belief("belief_bike", "The user commutes by bicycle."),
        candidate_replacement_beliefs=(),
        candidate_conditions=(),
        temporal_context=(),
    )

    assert [edge.edge_type for edge in result] == [EvidenceEdgeType.REAFFIRMS]
    assert result[0].target_kind == "belief"
    assert result[0].target_id == "belief_bike"


def test_heuristic_uncertain_targets_belief() -> None:
    result = HeuristicEvidenceEdgeVerifier().verify_edges(
        new_evidence=evidence("evidence_unclear", "It is unclear whether the user still bikes to work."),
        candidate_belief=belief("belief_bike", "The user commutes by bicycle."),
        candidate_replacement_beliefs=(),
        candidate_conditions=(),
        temporal_context=(),
    )

    assert [edge.edge_type for edge in result] == [EvidenceEdgeType.UNCERTAIN]
    assert result[0].target_kind == "belief"
    assert result[0].target_id == "belief_bike"


def test_evidence_edge_verifier_protocol_signature() -> None:
    import inspect
    from retracemem.verifier.contracts import EvidenceEdgeVerifier
    sig = inspect.signature(EvidenceEdgeVerifier.verify_edges)
    params = list(sig.parameters.keys())
    assert "candidate_replacement_beliefs" in params
    assert params == [
        "self",
        "new_evidence",
        "candidate_belief",
        "candidate_replacement_beliefs",
        "candidate_conditions",
        "temporal_context",
    ]


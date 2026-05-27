from __future__ import annotations

from retracemem.schemas import EvidenceEdge, EvidenceEdgeType
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
        candidate_conditions=(condition("condition:user_1:mobility", "current mobility ability"),),
        temporal_context=(),
    )

    assert result == []


def test_heuristic_supersedes_targets_belief_and_includes_replacement() -> None:
    result = HeuristicEvidenceEdgeVerifier().verify_edges(
        new_evidence=evidence("evidence_address", "The user moved to 88 Cedar Avenue."),
        candidate_belief=belief("belief_address", "The user lives at 14 Pine Street."),
        candidate_conditions=(),
        temporal_context=(),
    )

    assert [edge.edge_type for edge in result] == [EvidenceEdgeType.SUPERSEDES]
    assert result[0].target_kind == "belief"
    assert result[0].target_id == "belief_address"
    assert result[0].replacement_belief_id


def test_heuristic_reaffirms_targets_belief() -> None:
    result = HeuristicEvidenceEdgeVerifier().verify_edges(
        new_evidence=evidence("evidence_still_bikes", "The user still commutes by bicycle."),
        candidate_belief=belief("belief_bike", "The user commutes by bicycle."),
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
        candidate_conditions=(),
        temporal_context=(),
    )

    assert [edge.edge_type for edge in result] == [EvidenceEdgeType.UNCERTAIN]
    assert result[0].target_kind == "belief"
    assert result[0].target_id == "belief_bike"

from __future__ import annotations

from retracemem.schemas import DependencyEdge
from retracemem.verifier.requirement_inducer import (
    HeuristicRequirementInducer,
    ManualRequirementInducer,
)
from tests.verifier_contract._helpers import belief, dependency, evidence


def test_manual_requirement_inducer_returns_typed_dependency_edges_with_provenance() -> None:
    edge = dependency()
    inducer = ManualRequirementInducer((edge,))

    result = inducer.induce_requirements(
        belief("belief_bike", "The user commutes by bicycle."),
        (evidence("support_1", "The user said they commute by bicycle."),),
    )

    assert result == [edge]
    assert isinstance(result[0], DependencyEdge)
    assert result[0].edge_type == "REQUIRES"
    assert result[0].supporting_evidence_ids == ("support_1",)
    assert result[0].inducer == "manual_fixture"
    assert result[0].rationale


def test_heuristic_requirement_inducer_emits_only_dependency_edges() -> None:
    inducer = HeuristicRequirementInducer()

    result = inducer.induce_requirements(
        belief("belief_bike", "The user commutes by bicycle."),
        (evidence("support_1", "The user said they commute by bicycle."),),
    )

    assert result
    assert all(isinstance(edge, DependencyEdge) for edge in result)
    assert {edge.edge_type for edge in result} == {"REQUIRES"}
    assert all(edge.belief_id == "belief_bike" for edge in result)
    assert all(edge.supporting_evidence_ids for edge in result)
    assert not any(hasattr(edge, "authorization_status") for edge in result)

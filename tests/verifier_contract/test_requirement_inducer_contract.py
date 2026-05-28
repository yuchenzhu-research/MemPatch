from __future__ import annotations

import pytest

from retracemem.schemas import BeliefNode, DependencyEdge
from retracemem.verifier.contracts import RequirementProposal
from retracemem.verifier.requirement_inducer import (
    HeuristicRequirementInducer,
    ManualRequirementInducer,
)
from tests.verifier_contract._helpers import belief, condition, dependency, evidence


def test_manual_requirement_inducer_returns_proposals() -> None:
    cond = condition("condition:user_1:mobility", "Mobility requirement")
    edge = dependency()
    proposal = RequirementProposal(condition=cond, dependency_edge=edge)
    inducer = ManualRequirementInducer((proposal,))

    result = inducer.induce_requirements(
        belief("belief_bike", "The user commutes by bicycle."),
        (evidence("support_1", "The user said they commute by bicycle."),),
    )

    assert len(result) == 1
    assert isinstance(result[0], RequirementProposal)
    assert result[0].condition == cond
    assert result[0].dependency_edge == edge
    assert result[0].dependency_edge.edge_type == "REQUIRES"
    assert result[0].dependency_edge.supporting_evidence_ids == ("support_1",)
    assert result[0].dependency_edge.inducer == "manual_fixture"
    assert result[0].dependency_edge.rationale


def test_manual_requirement_inducer_validation_mismatched_condition_id() -> None:
    cond = condition("condition:user_1:availability", "Availability requirement")
    edge = dependency()  # condition_id is "condition:user_1:mobility"
    proposal = RequirementProposal(condition=cond, dependency_edge=edge)
    inducer = ManualRequirementInducer((proposal,))

    with pytest.raises(ValueError, match="Mismatched condition_id"):
        inducer.induce_requirements(
            belief("belief_bike", "The user commutes by bicycle."),
            (evidence("support_1", "The user said they commute by bicycle."),),
        )


def test_manual_requirement_inducer_validation_mismatched_belief_id() -> None:
    cond = condition("condition:user_1:mobility", "Mobility requirement")
    edge = dependency()  # belief_id is "belief_bike"
    proposal = RequirementProposal(condition=cond, dependency_edge=edge)
    inducer = ManualRequirementInducer()
    # Manually register proposal under a mismatched belief ID to trigger validation
    inducer._proposals_by_belief["belief_car"] = [proposal]

    with pytest.raises(ValueError, match="Mismatched belief_id"):
        inducer.induce_requirements(
            belief("belief_car", "The user drives a car."),
            (evidence("support_1", "The user said they drive a car."),),
        )


def test_heuristic_requirement_inducer_emits_proposals() -> None:
    inducer = HeuristicRequirementInducer()

    result = inducer.induce_requirements(
        belief("belief_bike", "The user commutes by bicycle."),
        (evidence("support_1", "The user said they commute by bicycle."),),
    )

    assert result
    assert all(isinstance(proposal, RequirementProposal) for proposal in result)
    assert all(proposal.dependency_edge.edge_type == "REQUIRES" for proposal in result)
    assert all(proposal.dependency_edge.belief_id == "belief_bike" for proposal in result)
    assert all(proposal.dependency_edge.supporting_evidence_ids == ("support_1",) for proposal in result)
    assert all(proposal.condition.scope_id == "user_1" for proposal in result)
    assert all(proposal.dependency_edge.condition_id == proposal.condition.condition_id for proposal in result)
    assert result[0].condition.text == "The user currently has sufficient mobility for this activity."
    assert result[0].dependency_edge.rationale == "Mobility-related beliefs require current mobility ability."


def test_heuristic_requirement_inducer_missing_scope_id_raises_value_error() -> None:
    inducer = HeuristicRequirementInducer()

    bad_belief = BeliefNode(
        belief_id="belief_bike",
        proposition="The user commutes by bicycle.",
        source_evidence_ids=("support_1",),
        metadata={},  # missing scope_id
    )

    with pytest.raises(ValueError, match="scope_id is missing in belief.metadata"):
        inducer.induce_requirements(
            bad_belief,
            (evidence("support_1", "The user said they commute by bicycle."),),
        )

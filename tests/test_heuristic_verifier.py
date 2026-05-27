from __future__ import annotations

from retracemem.schemas import Belief, EpisodicEvidence, RelationPrediction, RelationType
from retracemem.verifier import HeuristicRelationVerifier


def _verify(evidence_text: str, belief_text: str) -> RelationPrediction:
    verifier = HeuristicRelationVerifier()
    evidence = EpisodicEvidence(
        id="e1",
        timestamp="2026-05-27T00:00:00Z",
        text=evidence_text,
        source_id="test",
    )
    belief = Belief(id="b1", proposition=belief_text)
    return verifier.verify(evidence, belief)


def test_predicts_supersede_for_address_replacement() -> None:
    prediction = _verify(
        "The user moved to 88 Cedar Avenue last week.",
        "The user lives at 14 Pine Street.",
    )

    assert prediction.relation == RelationType.SUPERSEDE
    assert prediction.target_belief_id == "b1:replacement"


def test_predicts_block_for_injury_and_mobility_belief() -> None:
    prediction = _verify(
        "The user broke their leg and will be in a cast for six weeks.",
        "The user commutes by bicycle.",
    )

    assert prediction.relation == RelationType.BLOCK
    assert prediction.condition == "cycling ability"


def test_predicts_condition_for_explicit_condition() -> None:
    prediction = _verify(
        "The user can run only if their doctor clears them after the follow-up visit.",
        "The user can resume running workouts.",
    )

    assert prediction.relation == RelationType.CONDITION
    assert prediction.condition


def test_predicts_none_for_unrelated_topics() -> None:
    prediction = _verify(
        "The user tried a new Thai restaurant downtown.",
        "The user lives at 14 Pine Street.",
    )

    assert prediction.relation == RelationType.NONE


def test_predicts_uncertain_for_ambiguous_revision() -> None:
    prediction = _verify(
        "The user's address might have changed, but the new address is not confirmed.",
        "The user lives at 14 Pine Street.",
    )

    assert prediction.relation == RelationType.UNCERTAIN


def test_predicts_support_for_repeated_support_language() -> None:
    prediction = _verify(
        "The user still commutes by bicycle on weekdays.",
        "The user commutes by bicycle.",
    )

    assert prediction.relation == RelationType.SUPPORT


def test_empty_input_fails_closed_without_crashing() -> None:
    prediction = HeuristicRelationVerifier().verify("", "")

    assert isinstance(prediction, RelationPrediction)
    assert prediction.relation == RelationType.UNCERTAIN
    assert prediction.confidence == 0.0


def test_output_schema_fields_are_populated() -> None:
    prediction = _verify(
        "The user still commutes by bicycle on weekdays.",
        "The user commutes by bicycle.",
    )

    assert isinstance(prediction, RelationPrediction)
    assert prediction.evidence_id == "e1"
    assert prediction.belief_id == "b1"
    assert prediction.rationale
    assert prediction.span
    assert isinstance(prediction.confidence, float)

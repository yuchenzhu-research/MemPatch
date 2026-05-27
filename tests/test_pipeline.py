from __future__ import annotations

from pathlib import Path

from retracemem.evaluation.jsonl import read_jsonl, write_jsonl
from retracemem.pipeline import ReTracePipeline
from retracemem.schemas import Belief, EpisodicEvidence, RelationPrediction, RelationType


class StaticVerifier:
    def __init__(self, predictions: dict[str, RelationPrediction]) -> None:
        self.predictions = predictions

    def verify(
        self,
        new_evidence: EpisodicEvidence,
        candidate_belief: Belief,
        context: dict[str, object] | None = None,
    ) -> RelationPrediction:
        del context
        return self.predictions.get(
            candidate_belief.id,
            RelationPrediction(
                relation=RelationType.NONE,
                evidence_id=new_evidence.id,
                belief_id=candidate_belief.id,
            ),
        )


def _belief(belief_id: str, proposition: str) -> Belief:
    return Belief(id=belief_id, proposition=proposition, supported_by=[f"support_{belief_id}"])


def _evidence(evidence_id: str, text: str) -> EpisodicEvidence:
    return EpisodicEvidence(
        id=evidence_id,
        timestamp="2026-05-27T09:00:00Z",
        text=text,
        source_id="test_session",
    )


def test_broken_leg_evidence_blocks_bike_commute() -> None:
    pipeline = ReTracePipeline(
        verifier=StaticVerifier(
            {
                "belief_bike": RelationPrediction(
                    relation=RelationType.BLOCK,
                    condition="cycling ability",
                )
            }
        )
    )
    pipeline.add_belief("user_1", _belief("belief_bike", "The user usually commutes by bicycle."))

    accepted = pipeline.ingest_evidence(
        "user_1",
        _evidence("episode_broken_leg", "The user broke their leg and will be in a cast for six weeks."),
    )
    basis = pipeline.authorized_basis("user_1", "How should the user commute tomorrow?")

    assert [relation.relation for relation in accepted] == [RelationType.BLOCK]
    assert basis == []


def test_unrelated_protected_belief_is_preserved() -> None:
    pipeline = ReTracePipeline(
        verifier=StaticVerifier(
            {
                "belief_bike": RelationPrediction(
                    relation=RelationType.BLOCK,
                    condition="cycling ability",
                )
            }
        )
    )
    pipeline.add_belief("user_1", _belief("belief_bike", "The user usually commutes by bicycle."))
    pipeline.add_belief("user_1", _belief("belief_food", "The user likes Thai food."))

    pipeline.ingest_evidence(
        "user_1",
        _evidence("episode_broken_leg", "The user broke their leg and will be in a cast for six weeks."),
    )
    basis = pipeline.authorized_basis("user_1", "What should we remember about the user?")

    assert {item["belief_id"] for item in basis} == {"belief_food"}
    assert basis[0]["text"] == "The user likes Thai food."


def test_supersede_old_address_adds_and_authorizes_replacement() -> None:
    pipeline = ReTracePipeline(
        verifier=StaticVerifier(
            {
                "belief_old_address": RelationPrediction(
                    relation=RelationType.SUPERSEDE,
                    target_belief_id="belief_new_address",
                )
            }
        )
    )
    pipeline.add_belief("user_1", _belief("belief_old_address", "The user lives at 12 Pine Street."))

    pipeline.ingest_evidence(
        "user_1",
        _evidence("episode_address", "The user now lives at 48 Cedar Avenue."),
    )
    basis = pipeline.authorized_basis("user_1", "Where does the user live?")

    assert {item["belief_id"] for item in basis} == {"belief_new_address"}
    assert basis[0]["text"] == "The user now lives at 48 Cedar Avenue."


def test_uncertain_evidence_removes_default_without_replacement() -> None:
    pipeline = ReTracePipeline(
        verifier=StaticVerifier(
            {
                "belief_address": RelationPrediction(
                    relation=RelationType.UNCERTAIN,
                    rationale="The latest evidence makes the address unclear.",
                )
            }
        )
    )
    pipeline.add_belief("user_1", _belief("belief_address", "The user lives at 12 Pine Street."))

    pipeline.ingest_evidence(
        "user_1",
        _evidence("episode_unclear", "The user's address might have changed, but it is unclear."),
    )
    record = pipeline.answer("user_1", "Where does the user live?")

    assert record.authorized_basis == []
    assert [item["belief_id"] for item in record.blocked_beliefs] == ["belief_address"]
    assert [item["belief_id"] for item in record.candidate_beliefs] == ["belief_address"]


def test_answer_record_is_jsonl_compatible(tmp_path: Path) -> None:
    pipeline = ReTracePipeline(verifier=StaticVerifier({}))
    pipeline.add_belief("user_1", _belief("belief_food", "The user likes Thai food."))

    record = pipeline.answer("user_1", "What food does the user like?")
    output_path = tmp_path / "answers.jsonl"
    write_jsonl([record], output_path)

    loaded = read_jsonl(output_path)

    assert loaded[0]["query_id"] == "user_1:What food does the user like?"
    assert loaded[0]["method"] == "retrace_pipeline"
    assert loaded[0]["authorized_basis"] == [
        {"belief_id": "belief_food", "text": "The user likes Thai food."}
    ]
    assert loaded[0]["blocked_beliefs"] == []

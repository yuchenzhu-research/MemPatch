from retracemem.memory.belief_store import BeliefStore
from retracemem.memory.episode_ledger import EpisodeLedger
from retracemem.schemas import Belief, EpisodicEvidence, RelationPrediction, RelationType


def test_episode_ledger_is_append_only_and_rejects_duplicate_ids() -> None:
    ledger = EpisodeLedger()
    evidence = EpisodicEvidence(
        id="episode_1",
        timestamp="2026-05-27T09:00:00Z",
        text="The user broke their leg.",
        source_id="session_1",
    )

    ledger.append(evidence)

    assert ledger.get("episode_1") == evidence
    assert ledger.ids() == ["episode_1"]
    assert "episode_1" in ledger
    assert len(ledger) == 1
    try:
        ledger.append(evidence)
    except ValueError as exc:
        assert "evidence already exists" in str(exc)
    else:
        raise AssertionError("duplicate evidence id should fail")


def test_belief_store_rejects_duplicate_belief_ids_and_keeps_relations_local() -> None:
    store = BeliefStore()
    belief = Belief(id="belief_bike", proposition="The user commutes by bicycle.")
    relation = RelationPrediction(
        relation=RelationType.BLOCK,
        evidence_id="episode_broken_leg",
        belief_id=belief.id,
        condition="cycling ability",
    )

    store.add_belief(belief)
    store.add_relation(relation)

    assert store.get_belief(belief.id) == belief
    assert store.has_belief(belief.id)
    assert store.relations_for_belief(belief.id) == [relation]
    assert len(store) == 1
    try:
        store.add_belief(belief)
    except ValueError as exc:
        assert "belief already exists" in str(exc)
    else:
        raise AssertionError("duplicate belief id should fail")

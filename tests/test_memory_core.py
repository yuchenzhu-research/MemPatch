from retracemem.memory.belief_store import BeliefStore
from retracemem.memory.episode_ledger import EpisodeLedger
from retracemem.schemas import BeliefNode, EvidenceNode


def test_episode_ledger_is_append_only_and_rejects_duplicate_ids() -> None:
    ledger = EpisodeLedger()
    evidence = EvidenceNode(
        evidence_id="episode_1",
        session_id="session_1",
        timestamp="2026-05-27T09:00:00Z",
        text="The user broke their leg.",
        source_dataset="manual",
        source_pointer="test",
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


def test_belief_store_rejects_duplicate_belief_ids() -> None:
    store = BeliefStore()
    belief = BeliefNode(
        belief_id="belief_bike",
        proposition="The user commutes by bicycle.",
        source_evidence_ids=("episode_1",),
    )

    store.add_belief(belief)

    assert store.get_belief("belief_bike") == belief
    assert store.has_belief("belief_bike")
    assert len(store) == 1
    try:
        store.add_belief(belief)
    except ValueError as exc:
        assert "belief already exists" in str(exc)
    else:
        raise AssertionError("duplicate belief id should fail")

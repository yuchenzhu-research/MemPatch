from retracemem.memory.belief_store import BeliefStore
from retracemem.generation.basis_builder import BasisBuilder
from retracemem.schemas import Belief, RelationPrediction, RelationType
from retracemem.tms.authorization import AuthorizationEngine
from retracemem.tms.gate import RevisionGate


def _belief(belief_id: str, proposition: str) -> Belief:
    return Belief(id=belief_id, proposition=proposition, supported_by=[f"support_{belief_id}"])


def test_gate_accepts_block_only_with_condition() -> None:
    gate = RevisionGate()

    assert gate.accept_local_relation(
        RelationPrediction(
            relation=RelationType.BLOCK,
            evidence_id="e1",
            belief_id="b1",
            condition="cycling ability",
        )
    )
    assert not gate.accept_local_relation(
        RelationPrediction(relation=RelationType.BLOCK, evidence_id="e1", belief_id="b1")
    )


def test_gate_accepts_block_with_explicit_prerequisite() -> None:
    gate = RevisionGate()

    assert gate.accept_local_relation(
        RelationPrediction(
            relation=RelationType.BLOCK,
            evidence_id="e1",
            belief_id="bike_commute",
            target_belief_id="can_ride_bike",
        )
    )


def test_gate_rejects_none_as_revision_operation() -> None:
    gate = RevisionGate()

    assert not gate.accept_local_relation(
        RelationPrediction(relation=RelationType.NONE, evidence_id="e1", belief_id="b1")
    )


def test_authorization_blocks_belief_with_defeat_relation() -> None:
    store = BeliefStore()
    belief = Belief(
        id="belief_bike_commute",
        proposition="The user commonly commutes by bicycle.",
        supported_by=["session_002_span_01"],
    )
    store.add_belief(belief)
    store.add_relation(
        RelationPrediction(
            relation=RelationType.BLOCK,
            evidence_id="session_007_span_02",
            belief_id=belief.id,
            condition="cycling ability",
        )
    )

    decision = AuthorizationEngine(store).decide(belief)

    assert not decision.authorized
    assert decision.reason == "blocked"


def test_broken_leg_blocks_bike_commute_but_not_food_preference() -> None:
    store = BeliefStore()
    bike_commute = _belief("belief_bike_commute", "The user usually commutes by bicycle.")
    food_preference = _belief("belief_food_preference", "The user likes spicy noodles.")
    store.add_belief(bike_commute)
    store.add_belief(food_preference)
    store.add_relation(
        RelationPrediction(
            relation=RelationType.BLOCK,
            evidence_id="episode_broken_leg",
            belief_id=bike_commute.id,
            condition="cycling ability",
            rationale="A broken leg prevents bicycle commuting while it applies.",
        )
    )

    engine = AuthorizationEngine(store)
    bike_decision = engine.decide(bike_commute)
    food_decision = engine.decide(food_preference)
    basis = BasisBuilder(store).build("How does the user commute and what food do they like?")

    assert not bike_decision.authorized
    assert bike_decision.reason == "blocked"
    assert food_decision.authorized
    assert food_decision.reason == "supported"
    assert {item["belief_id"] for item in basis} == {food_preference.id}


def test_invalid_block_relation_does_not_block_belief() -> None:
    store = BeliefStore()
    belief = _belief("belief_bike_commute", "The user usually commutes by bicycle.")
    store.add_belief(belief)
    store.add_relation(
        RelationPrediction(
            relation=RelationType.BLOCK,
            evidence_id="episode_broken_leg",
            belief_id=belief.id,
        )
    )

    decision = AuthorizationEngine(store).decide(belief)

    assert decision.authorized
    assert decision.reason == "supported"


def test_supersede_blocks_old_belief_and_preserves_new_belief() -> None:
    store = BeliefStore()
    old_address = _belief("belief_old_address", "The user lives at 12 Pine Street.")
    new_address = _belief("belief_new_address", "The user lives at 48 Cedar Avenue.")
    store.add_belief(old_address)
    store.add_belief(new_address)
    store.add_relation(
        RelationPrediction(
            relation=RelationType.SUPERSEDE,
            evidence_id="episode_address_update",
            belief_id=old_address.id,
            target_belief_id=new_address.id,
        )
    )

    engine = AuthorizationEngine(store)
    old_decision = engine.decide(old_address)
    new_decision = engine.decide(new_address)
    basis = BasisBuilder(store).build("Where does the user live?")

    assert not old_decision.authorized
    assert old_decision.reason == "superseded"
    assert new_decision.authorized
    assert {item["belief_id"] for item in basis} == {new_address.id}


def test_uncertain_relation_removes_default_without_creating_replacement() -> None:
    store = BeliefStore()
    address = _belief("belief_address", "The user lives at 12 Pine Street.")
    store.add_belief(address)
    store.add_relation(
        RelationPrediction(
            relation=RelationType.UNCERTAIN,
            evidence_id="episode_unclear_address",
            belief_id=address.id,
            rationale="The latest evidence makes the current address uncertain.",
        )
    )
    belief_ids_before = {belief.id for belief in store.all_beliefs()}

    decision = AuthorizationEngine(store).decide(address)
    basis = BasisBuilder(store).build("Where does the user live?")
    belief_ids_after = {belief.id for belief in store.all_beliefs()}

    assert not decision.authorized
    assert decision.reason == "uncertain"
    assert basis == []
    assert belief_ids_after == belief_ids_before == {address.id}


def test_toy_revision_cases_from_fixture() -> None:
    import json
    import os
    from retracemem.memory.episode_ledger import EpisodeLedger
    from retracemem.schemas import EpisodicEvidence

    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "toy_revision_cases.jsonl")
    with open(fixture_path, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    for case in cases:
        category = case["category"]
        ledger = EpisodeLedger()
        store = BeliefStore()

        for ev_data in case["evidences"]:
            ev = EpisodicEvidence(
                id=ev_data["id"],
                timestamp=ev_data["timestamp"],
                text=ev_data["text"],
                source_id=ev_data["source_id"],
            )
            ledger.append(ev)

        for b_data in case["beliefs"]:
            b = Belief(
                id=b_data["id"],
                proposition=b_data["proposition"],
                supported_by=b_data["supported_by"],
            )
            store.add_belief(b)

        for r_data in case["relations"]:
            rel_type = RelationType(r_data["relation"])
            rel = RelationPrediction(
                relation=rel_type,
                evidence_id=r_data.get("evidence_id"),
                belief_id=r_data.get("belief_id"),
                target_belief_id=r_data.get("target_belief_id"),
                condition=r_data.get("condition"),
            )
            store.add_relation(rel)

        engine = AuthorizationEngine(store, ledger)
        for cutoff_ev, expected_states in case["expected"].items():
            for belief_id, expected_auth in expected_states.items():
                belief = store.get_belief(belief_id)
                dec = engine.decide(belief, at_evidence_id=cutoff_ev)
                assert dec.authorized == expected_auth, (
                    f"Category {category} failed: at cutoff {cutoff_ev}, "
                    f"belief {belief_id} expected authorized={expected_auth}, got {dec.authorized}"
                )


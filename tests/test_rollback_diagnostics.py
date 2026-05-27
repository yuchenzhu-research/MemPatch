from __future__ import annotations

import json
import os

from retracemem.memory.belief_store import BeliefStore
from retracemem.memory.episode_ledger import EpisodeLedger
from retracemem.schemas import Belief, EpisodicEvidence, RelationPrediction, RelationType
from retracemem.tms.rollback import RollbackDiagnostics


def test_rollback_diagnostics_calculation() -> None:
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "toy_revision_cases.jsonl")
    with open(fixture_path, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    # Find the rollback case
    rollback_case = None
    for case in cases:
        if case["category"] == "rollback":
            rollback_case = case
            break

    assert rollback_case is not None

    ledger = EpisodeLedger()
    store = BeliefStore()

    for ev_data in rollback_case["evidences"]:
        ev = EpisodicEvidence(
            id=ev_data["id"],
            timestamp=ev_data["timestamp"],
            text=ev_data["text"],
            source_id=ev_data["source_id"],
        )
        ledger.append(ev)

    for b_data in rollback_case["beliefs"]:
        b = Belief(
            id=b_data["id"],
            proposition=b_data["proposition"],
            supported_by=b_data["supported_by"],
        )
        store.add_belief(b)

    for r_data in rollback_case["relations"]:
        rel_type = RelationType(r_data["relation"])
        rel = RelationPrediction(
            relation=rel_type,
            evidence_id=r_data.get("evidence_id"),
            belief_id=r_data.get("belief_id"),
            target_belief_id=r_data.get("target_belief_id"),
            condition=r_data.get("condition"),
        )
        store.add_relation(rel)

    diag = RollbackDiagnostics(store, ledger)

    # 1. HER test
    # If total inputs was 3, and we retained 3
    her = diag.calculate_her(3)
    assert her == 1.0

    # 2. RAR and URR test
    # Expected dict matches the expected transitions.
    # ground_truth maps evidence_id -> {belief_id: expected_authorized}
    ground_truth = rollback_case["expected"]

    metrics = diag.calculate_rar_and_urr(ground_truth)
    assert "RAR" in metrics
    assert "URR_react" in metrics
    # Under standard rollback: b_commute_bike: True -> False -> True
    # The recovery transition from ev_broke_leg (False) to ev_recovered (True)
    # Expected RAR is 1.0, URR_react is 0.0
    assert metrics["RAR"] == 1.0
    assert metrics["URR_react"] == 0.0

"""Archived: Stage A failure-analysis diagnostic over a parsed run directory.

Relocated from tests/experiments/stageab/test_constrained_proposer_and_diagnostics.py
because it exercises the archived diagnostic
experiments.archive.legacy.analyze_stagea_failures, which is not part of the
canonical Stage A/B/C evaluation path. The diagnostic still relies on the active
metric computation (retracemem.evaluation.multiagent.metrics).
"""
from __future__ import annotations

import json
import sys

import pytest

from retracemem.schemas import (
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceNode,
)
from retracemem.evaluation.multiagent.contracts import (
    FixedCandidateGoldRecord,
    FixedCandidateInputEpisode,
    FixedCandidateSubmission,
    GoldSnapshotExpectation,
    TypedRevisionTarget,
)


@pytest.fixture
def tiny_submission():
    ev = EvidenceNode("ev_1", "sess_1", "2026-05-30T00:00:00Z", "Some evidence", "dataset", "pointer")
    b = BeliefNode("b_1", "Proposition 1", ("ev_1",))
    b2 = BeliefNode("b_2", "Proposition 2", ("ev_1",))
    c = ConditionNode("c_1", "scope_1", "Condition 1")
    dep = DependencyEdge("dep_1", "b_1", "c_1", "system")

    return FixedCandidateSubmission(
        submission_id="sub_1",
        producer_id="writer",
        producer_role="writer",
        task_id="task_1",
        parent_snapshot_id="snapshot_init",
        observed_at="2026-05-30T00:00:00Z",
        instance_id="inst_1",
        query_id="q_1",
        query="Check status?",
        evidence_context=(ev,),
        new_evidence_id="ev_1",
        candidate_beliefs=(b,),
        candidate_replacement_beliefs=(b2,),
        candidate_conditions_by_belief=(("b_1", (c,)),),
        dependency_edges_by_belief=(("b_1", (dep,)),),
    )


def test_action_confusion_diagnostics_on_tiny_fixture(tmp_path, tiny_submission):
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()

    row = {
        "episode_id": "ep_test",
        "submissions": [{
            "submission_id": "sub_1",
            "actions": [{
                "action_type": "BLOCKS",
                "target_belief_id": None,
                "target_condition_id": "c_1",
                "replacement_belief_id": None,
                "rationale": "mock block",
                "evidence_ids": ["ev_1"]
            }],
            "proposal_edges": [],
            "parse_error": None
        }],
        "final_belief_statuses": {
            "b_1": "BLOCKED"
        }
    }

    stage_a_parsed_file = run_dir / "stage_a_parsed.jsonl"
    with open(stage_a_parsed_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    gold_snapshot = GoldSnapshotExpectation(
        belief_statuses={"b_1": "BLOCKED"}
    )
    gold = FixedCandidateGoldRecord(
        episode_id="ep_test",
        gold_snapshot=gold_snapshot,
        gold_typed_targets=(
            TypedRevisionTarget("sub_1", "BLOCKS", target_condition_id="c_1", evidence_ids=("ev_1",)),
        ),
        failure_type="stale_propagation",
    )

    episode = FixedCandidateInputEpisode(
        episode_id="ep_test",
        domain="software_engineering",
        failure_type_public_or_controlled="stale_propagation",
        subagent_roles=("writer",),
        submissions=(tiny_submission,),
        downstream_tasks=(),
    )

    import experiments.archive.legacy.analyze_stagea_failures as analyze_mod

    analyze_mod.generate_expanded_episodes = lambda: ((episode, gold),)

    sys.argv = ["analyze_stagea_failures.py", "--run-dir", str(run_dir)]
    analyze_mod.main()

    assert (run_dir / "summary.json").exists()
    assert (run_dir / "action_confusion.csv").exists()
    assert (run_dir / "evidence_grounding_errors.csv").exists()
    assert (run_dir / "no_revision_bias.csv").exists()
    assert (run_dir / "per_failure_type_metrics.csv").exists()
    assert (run_dir / "representative_failures.md").exists()

    with open(run_dir / "summary.json", "r") as f:
        summary = json.load(f)
        assert summary["total_cases"] == 1
        assert summary["final_status_accuracy"] == 1.0
        assert summary["exact_action_match"] == 1.0

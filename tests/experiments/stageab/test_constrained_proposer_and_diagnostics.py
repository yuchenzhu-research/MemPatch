from __future__ import annotations

import json
import pytest
from retracemem.schemas import (
    EvidenceNode,
    BeliefNode,
    ConditionNode,
    DependencyEdge,
)
from experiments.multiagent.contracts import (
    FixedCandidateSubmission,
    TypedRevisionTarget,
    FixedCandidateGoldRecord,
    FixedCandidateInputEpisode,
    GoldSnapshotExpectation,
)
from retracemem.multiagent.utils import build_candidate_actions
from experiments.multiagent.stagec_policy import (
    ClosedAPIZeroShotConstrainedProposer,
    PromptTypedRevisionPolicy,
)


@pytest.fixture
def tiny_submission():
    ev = EvidenceNode("ev_1", "sess_1", "2026-05-30T00:00:00Z", "Some evidence", "dataset", "pointer")
    b = BeliefNode("b_1", "Proposition 1", ("ev_1",))
    b2 = BeliefNode("b_2", "Proposition 2", ("ev_1",))
    c = ConditionNode("c_1", "scope_1", "Condition 1")
    dep = DependencyEdge("dep_1", "b_1", "c_1", "system")

    sub = FixedCandidateSubmission(
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
    return sub


def test_candidate_affordance_builder_never_uses_gold(tiny_submission):
    # The submission object has absolutely no gold expectations or label assignments.
    # The builder must succeed without raising errors.
    candidates = build_candidate_actions(tiny_submission)
    assert len(candidates) > 0


def test_supersedes_candidate_exists_when_replacement_beliefs_visible(tiny_submission):
    candidates = build_candidate_actions(tiny_submission)
    supersedes = [c for c in candidates if c["action_type"] == "SUPERSEDES"]
    assert len(supersedes) == 1
    assert supersedes[0]["target_belief_id"] == "b_1"
    assert supersedes[0]["replacement_belief_id"] == "b_2"


def test_blocks_releases_candidates_exist_when_conditions_visible(tiny_submission):
    candidates = build_candidate_actions(tiny_submission)
    blocks = [c for c in candidates if c["action_type"] == "BLOCKS"]
    releases = [c for c in candidates if c["action_type"] == "RELEASES"]
    
    assert len(blocks) == 1
    assert blocks[0]["target_condition_id"] == "c_1"
    assert len(releases) == 1
    assert releases[0]["target_condition_id"] == "c_1"


def test_no_revision_fallback_exists(tiny_submission):
    candidates = build_candidate_actions(tiny_submission)
    no_rev = [c for c in candidates if c["action_type"] == "NO_REVISION"]
    assert len(no_rev) == 1
    assert no_rev[0]["candidate_action_id"] == "act_no_revision"


def test_constrained_parser_rejects_invented_candidate_ids(tiny_submission):
    proposer = ClosedAPIZeroShotConstrainedProposer()
    candidates = build_candidate_actions(tiny_submission)
    
    # Invented ID "act_invented" not present in candidates
    invalid_response = json.dumps({
        "selected_candidate_action_ids": ["act_invented"],
        "rejection_reasons": {}
    })
    
    out = proposer.parse_response(
        invalid_response,
        example_id="ex_test",
        submission=tiny_submission,
        candidates=candidates
    )
    assert out.parsing_valid is False
    assert any("Invented candidate action ID" in err for err in out.errors)


def test_no_revision_cannot_combine_with_other_actions(tiny_submission):
    proposer = ClosedAPIZeroShotConstrainedProposer()
    candidates = build_candidate_actions(tiny_submission)
    
    # Combining act_no_revision and a real action ID
    invalid_response = json.dumps({
        "selected_candidate_action_ids": ["act_no_revision", "act_blocks_c_1"],
        "rejection_reasons": {}
    })
    
    out = proposer.parse_response(
        invalid_response,
        example_id="ex_test",
        submission=tiny_submission,
        candidates=candidates
    )
    assert out.parsing_valid is False
    assert any("NO_REVISION cannot be combined" in err for err in out.errors)


def test_constrained_selected_ids_map_back_to_typed_actions(tiny_submission):
    proposer = ClosedAPIZeroShotConstrainedProposer()
    candidates = build_candidate_actions(tiny_submission)
    
    valid_response = json.dumps({
        "selected_candidate_action_ids": ["act_blocks_c_1"],
        "rejection_reasons": {}
    })
    
    out = proposer.parse_response(
        valid_response,
        example_id="ex_test",
        submission=tiny_submission,
        candidates=candidates
    )
    assert out.parsing_valid is True
    assert len(out.parsed_actions) == 1
    assert out.parsed_actions[0].action_type == "BLOCKS"
    assert out.parsed_actions[0].target_condition_id == "c_1"
    assert len(out.proposal_batches) == 1
    assert len(out.proposal_batches[0].edges) == 1


def test_decision_audit_saved_but_not_passed_to_dpa(tiny_submission):
    # 1. Test standard policy in diagnostic mode
    policy = PromptTypedRevisionPolicy(diagnostic_mode=True)
    
    audit_data = {
        "new_evidence_role": "test role",
        "prior_replacement_relation": "test relation",
        "condition_effect": "test effect",
        "conflict_state": "test conflict",
        "selected_action_types": ["BLOCKS"],
        "rejected_action_types": {}
    }
    
    response = json.dumps({
        "decision_audit": audit_data,
        "actions": [{
            "action_type": "BLOCKS",
            "target_belief_id": None,
            "target_condition_id": "c_1",
            "replacement_belief_id": None,
            "rationale": "test block",
            "evidence_ids": ["ev_1"]
        }]
    })
    
    out = policy.parse_response(response, example_id="ex_test", submission=tiny_submission)
    assert out.parsing_valid is True
    assert out.metadata.get("decision_audit") == audit_data
    # DPA edges must only come from actions
    assert len(out.proposal_batches) == 1
    assert len(out.proposal_batches[0].edges) == 1
    assert out.proposal_batches[0].edges[0].target_id == "c_1"


def test_action_confusion_diagnostics_on_tiny_fixture(tmp_path, tiny_submission):
    # Setup a mock Stage A parsed run directory
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    
    # We construct mock run record matching stage_a_parsed.jsonl
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
        
    # We mock generate_expanded_episodes using monkeypatch
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
    
    import experiments.multiagent.legacy.analyze_stagea_failures as analyze_mod
    
    # Monkeypatch generate_expanded_episodes in the module
    analyze_mod.generate_expanded_episodes = lambda: ((episode, gold),)
    
    # Run the main analysis function by mocking argv
    import sys
    sys.argv = ["analyze_stagea_failures.py", "--run-dir", str(run_dir)]
    analyze_mod.main()
    
    # Verify outputs
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

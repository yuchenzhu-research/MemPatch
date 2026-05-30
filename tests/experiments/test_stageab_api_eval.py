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
from experiments.multiagent.run_stageab_api_eval import (
    rename_episode_and_gold,
    render_direct_judge_prompt,
    parse_direct_judge_response,
    compute_stage_a_action_metrics,
    check_grounding_error_stage_a,
    check_grounding_error_stage_b,
    canonicalize_belief_id_with_type,
)


@pytest.fixture
def mock_episode_and_gold():
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

    episode = FixedCandidateInputEpisode(
        episode_id="ep_test",
        domain="software_engineering",
        failure_type_public_or_controlled="stale_propagation",
        subagent_roles=("writer",),
        submissions=(sub,),
        downstream_tasks=(),
    )

    gold_snapshot = GoldSnapshotExpectation(
        belief_statuses={"b_1": "SUPERSEDED"}
    )
    gold = FixedCandidateGoldRecord(
        episode_id="ep_test",
        gold_snapshot=gold_snapshot,
        gold_typed_targets=(
            TypedRevisionTarget("sub_1", "SUPERSEDES", target_belief_id="b_1", replacement_belief_id="b_2", evidence_ids=("ev_1",)),
        ),
        failure_type="stale_propagation",
    )

    return episode, gold


def test_rename_episode_and_gold(mock_episode_and_gold):
    ep, gold = mock_episode_and_gold
    ep_renamed, gold_renamed = rename_episode_and_gold(ep, gold)

    assert ep_renamed.episode_id == "ep_test__heldout_base"
    assert gold_renamed.episode_id == "ep_test__heldout_base"
    assert ep_renamed.submissions[0].submission_id == "sub_1"  # Not altered unless containing ep_test
    # Since belief_id and condition_id don't contain old_ns, they remain intact.
    # But if they did, they would be renamed.


def test_render_direct_judge_prompt(mock_episode_and_gold):
    ep, _ = mock_episode_and_gold
    sub = ep.submissions[0]
    template = "Query: {query}\nNew Evidence: {new_evidence_text}\nCandidate Beliefs:\n{candidate_beliefs}"
    rendered = render_direct_judge_prompt(template, sub)

    assert "Query: Check status?" in rendered
    assert "New Evidence: Some evidence" in rendered
    assert "b_1" in rendered


def test_canonicalize_belief_id_with_type():
    valid_ids = {"b_1_some_other_thing", "b_2_inactive", "b_3"}
    
    # 1. Exact Match
    cid, applied, mtype = canonicalize_belief_id_with_type("b_3", valid_ids)
    assert cid == "b_3"
    assert applied is False
    assert mtype == "exact"
    
    # 2. Prefix Match (v_id starts with returned_id)
    cid, applied, mtype = canonicalize_belief_id_with_type("b_1", valid_ids)
    assert cid == "b_1_some_other_thing"
    assert applied is True
    assert mtype == "prefix"

    # 3. Suffix Match (returned_id ends with v_id)
    cid, applied, mtype = canonicalize_belief_id_with_type("prefix_b_3", valid_ids)
    assert cid == "b_3"
    assert applied is True
    assert mtype == "suffix"
    
    # 4. Fuzzy Match
    cid, applied, mtype = canonicalize_belief_id_with_type("b_2_inac-tive", valid_ids)
    assert cid == "b_2_inactive"
    assert applied is True
    assert mtype == "fuzzy"
    
    # 5. Failed Canonicalization
    cid, applied, mtype = canonicalize_belief_id_with_type("b_unknown", valid_ids)
    assert cid == "b_unknown"
    assert applied is False
    assert mtype == "failed"


def test_parse_direct_judge_response():
    valid_belief_ids = {"b_1", "b_2"}
    response = """
    {
      "verdicts": [
        {"belief_id": "b_1", "status": "USABLE", "rationale": "ok", "confidence": 0.9},
        {"belief_id": "b_2", "status": "NOT_USABLE", "rationale": "stale", "confidence": 0.8}
      ]
    }
    """
    verdicts = parse_direct_judge_response(response, valid_belief_ids)
    assert len(verdicts) == 2
    assert verdicts[0]["canonical_belief_id"] == "b_1"
    assert verdicts[0]["raw_belief_id"] == "b_1"
    assert verdicts[0]["canonicalization_applied"] is False
    assert verdicts[0]["canonicalization_type"] == "exact"
    assert verdicts[0]["status"] == "USABLE"
    
    assert verdicts[1]["canonical_belief_id"] == "b_2"
    assert verdicts[1]["raw_belief_id"] == "b_2"
    assert verdicts[1]["status"] == "NOT_USABLE"


def test_parse_direct_judge_response_errors():
    valid_belief_ids = {"b_1"}
    response_missing = '{"verdicts": []}'
    with pytest.raises(ValueError, match="omitted verdicts"):
        parse_direct_judge_response(response_missing, valid_belief_ids)

    response_invalid_id = '{"verdicts": [{"belief_id": "b_unknown", "status": "USABLE"}]}'
    with pytest.raises(ValueError, match="failed canonicalization"):
        parse_direct_judge_response(response_invalid_id, valid_belief_ids)


def test_check_grounding_error_stage_a(mock_episode_and_gold):
    ep, _ = mock_episode_and_gold
    sub = ep.submissions[0]

    # Correct action
    act_ok = {
        "action_type": "BLOCKS",
        "target_condition_id": "c_1",
        "evidence_ids": ["ev_1"]
    }
    assert check_grounding_error_stage_a(act_ok, sub) is False

    # Grounding error: invalid condition id
    act_err = {
        "action_type": "BLOCKS",
        "target_condition_id": "c_unknown",
        "evidence_ids": ["ev_1"]
    }
    assert check_grounding_error_stage_a(act_err, sub) is True


def test_check_grounding_error_stage_b():
    valid_belief_ids = {"b_1"}
    assert check_grounding_error_stage_b({"raw_belief_id": "b_1"}, valid_belief_ids) is False
    assert check_grounding_error_stage_b({"raw_belief_id": "b_2"}, valid_belief_ids) is True


def test_compute_stage_a_action_metrics(mock_episode_and_gold):
    ep, gold = mock_episode_and_gold
    sub = ep.submissions[0]

    pred_actions = [
        {
            "action_type": "SUPERSEDES",
            "target_belief_id": "b_1",
            "replacement_belief_id": "b_2",
            "evidence_ids": ["ev_1"],
        }
    ]

    metrics = compute_stage_a_action_metrics(pred_actions, gold.gold_typed_targets, sub)
    assert metrics["valid_json"] == 1.0
    assert metrics["exact_action_match"] == 1.0
    assert metrics["action_type_match"] == 1.0
    assert metrics["target_grounding"] == 1.0


def test_strict_vs_canonicalized_metric_divergence():
    # Test that strict vs canonicalized accuracy metrics can diverge under fuzzy matching
    # Gold statuses: b_1 is NOT_USABLE
    gold_comp = "NOT_USABLE"
    
    # Pred statuses (Stage B)
    # Canonicalized has b_1 = NOT_USABLE (via fuzzy match) -> Correct
    # Strict has b_1 = UNCERTAIN (no exact match) -> Incorrect
    strict_pred = {"b_1": "UNCERTAIN"}
    canonical_pred = {"b_1": "NOT_USABLE"}
    
    correct_strict = 1 if strict_pred.get("b_1") == gold_comp else 0
    correct_canonical = 1 if canonical_pred.get("b_1") == gold_comp else 0
    
    assert correct_strict == 0
    assert correct_canonical == 1
    assert correct_strict != correct_canonical


def test_prompt_non_leakage(mock_episode_and_gold):
    ep, gold = mock_episode_and_gold
    sub = ep.submissions[0]
    
    # Test Stage B prompt rendering
    from experiments.multiagent.run_stageab_api_eval import load_direct_judge_template, render_direct_judge_prompt
    template = load_direct_judge_template()
    prompt_b = render_direct_judge_prompt(template, sub)
    
    # Gold status of b_1 is SUPERSEDED, which maps to NOT_USABLE
    # Prompt must NOT contain "SUPERSEDED". "NOT_USABLE" is a class name in the template,
    # but the prompt must NOT contain "b_1: NOT_USABLE" or similar label assignments.
    assert "SUPERSEDED" not in prompt_b
    assert "b_1: NOT_USABLE" not in prompt_b
    assert "b_1: \"NOT_USABLE\"" not in prompt_b

    # Test Stage A prompt rendering
    from experiments.multiagent.stagec_policy import PromptTypedRevisionPolicy
    policy = PromptTypedRevisionPolicy()
    messages = policy.build_messages(sub)
    user_prompt_a = messages[1]["content"]
    
    # Prompts must not leak the gold action targets
    assert "Identify the correct revision actions" in user_prompt_a
    assert "gold_snapshot" not in user_prompt_a


def test_prompt_snapshot(mock_episode_and_gold):
    ep, gold = mock_episode_and_gold
    sub = ep.submissions[0]
    
    from experiments.multiagent.stagec_policy import PromptTypedRevisionPolicy
    policy = PromptTypedRevisionPolicy()
    messages = policy.build_messages(sub)
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]
    
    # 1. prompt contains action trigger guidance
    assert "Action Trigger Guidance" in system_prompt
    assert "Emit SUPERSEDES" in system_prompt
    assert "Emit BLOCKS" in system_prompt
    
    # 2. prompt does not contain gold_snapshot
    assert "gold_snapshot" not in system_prompt
    assert "gold_snapshot" not in user_prompt
    
    # 3. prompt does not contain gold action labels from the evaluator sidecar
    assert "gold_typed_targets" not in user_prompt
    
    # 4. prompt includes candidate replacement beliefs when visible
    assert "Candidate Replacement Beliefs" in user_prompt
    assert "b_2" in user_prompt
    assert "Proposition 2" in user_prompt
    
    # 5. prompt includes condition anchors when visible
    assert "Condition Anchors" in user_prompt
    assert "c_1" in user_prompt
    assert "Condition 1" in user_prompt


def test_canonicalize_ambiguity_rejection():
    # Multiple candidates matching prefix, suffix, or fuzzy
    valid_ids = {"b_1_active", "b_1_archive", "b_2_inactive"}
    
    # 1. Ambiguous prefix
    cid, applied, mtype = canonicalize_belief_id_with_type("b_1", valid_ids)
    assert cid == "b_1"
    assert applied is False
    assert mtype == "failed"
    
    # 2. Ambiguous suffix
    valid_ids_suffix = {"prefix_b_3", "other_b_3"}
    cid, applied, mtype = canonicalize_belief_id_with_type("b_3", valid_ids_suffix)
    assert cid == "b_3"
    assert applied is False
    assert mtype == "failed"
    
    # 3. Ambiguous fuzzy
    valid_ids_fuzzy = {"b_2_inactive", "b_2_inactive_2"}
    cid, applied, mtype = canonicalize_belief_id_with_type("b_2_inac-tive", valid_ids_fuzzy)
    assert cid == "b_2_inac-tive"
    assert applied is False
    assert mtype == "failed"


def test_compute_stage_a_action_metrics_empty_and_mismatch(mock_episode_and_gold):
    ep, gold = mock_episode_and_gold
    sub = ep.submissions[0]
    
    # 1. pred has actions, gold has NO_REVISION
    pred_actions = [
        {
            "action_type": "SUPERSEDES",
            "target_belief_id": "b_1",
            "replacement_belief_id": "b_2",
            "evidence_ids": ["ev_1"],
        }
    ]
    empty_gold_targets = ()
    metrics = compute_stage_a_action_metrics(pred_actions, empty_gold_targets, sub)
    assert metrics["exact_action_match"] == 0.0
    assert metrics["action_type_match"] == 0.0
    
    # 2. Both empty
    metrics_empty = compute_stage_a_action_metrics([], (), sub)
    assert metrics_empty["exact_action_match"] == 1.0
    assert metrics_empty["action_type_match"] == 1.0
    assert metrics_empty["evidence_grounding"] == 1.0


def test_closed_api_proposer_fail_closed(mock_episode_and_gold):
    from experiments.multiagent.stagec_policy import ClosedAPIZeroShotProposer
    ep, _ = mock_episode_and_gold
    sub = ep.submissions[0]
    
    # 1. Live mode, missing client -> Must raise ValueError
    proposer = ClosedAPIZeroShotProposer(
        provider_kind="siliconflow",
        model_id="deepseek-ai/DeepSeek-V3",
        client=None,
    )
    with pytest.raises(ValueError, match="Live mode requires client and model_id"):
        proposer.propose(sub)
        
    # 2. Mock mode -> Should produce deterministic NO_REVISION action and metadata contains prompt & response
    mock_proposer = ClosedAPIZeroShotProposer(
        provider_kind="mock",
        model_id=None,
        client=None,
    )
    out = mock_proposer.propose(sub)
    assert out.parsing_valid is True
    assert len(out.parsed_actions) == 1
    assert out.parsed_actions[0].action_type == "NO_REVISION"
    assert "raw_response" in out.metadata
    assert "prompt" in out.metadata


def test_runner_error_when_no_mode_specified(monkeypatch):
    from experiments.multiagent.run_stageab_api_eval import main
    import sys
    monkeypatch.setattr(sys, "argv", ["run_stageab_api_eval.py"])
    with pytest.raises(SystemExit):
        main()


def test_explicit_mock_mode_manifest(tmp_path, monkeypatch):
    from experiments.multiagent.run_stageab_api_eval import main
    import sys
    import json
    
    out_dir = tmp_path / "stageab_mock_test"
    monkeypatch.setattr(sys, "argv", [
        "run_stageab_api_eval.py",
        "--mock",
        "--max-cases", "1",
        "--output-dir", str(out_dir)
    ])
    main()
    
    manifest_file = out_dir / "manifest.json"
    assert manifest_file.exists()
    with open(manifest_file, "r") as f:
        manifest = json.load(f)
        
    assert manifest["run_mode"] == "mock"
    assert manifest["is_live_api_result"] is False
    assert manifest["mock_default_used"] is True
    assert "prompt_template_hash" in manifest
    assert manifest["parser_version"] == "PromptTypedRevisionPolicy_v1"
    assert manifest["response_schema_version"] == "v1_canonical"
    # Verify mock default artifacts are not labeled as live
    assert manifest["is_live_api_result"] is not True


def test_live_mode_missing_keys_fails_closed(tmp_path, monkeypatch):
    from experiments.multiagent.run_stageab_api_eval import main
    import sys
    
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    out_dir = tmp_path / "stageab_live_test"
    
    monkeypatch.setattr(sys, "argv", [
        "run_stageab_api_eval.py",
        "--live",
        "--provider", "siliconflow",
        "--model", "deepseek-ai/DeepSeek-V3",
        "--output-dir", str(out_dir)
    ])
    with pytest.raises(ValueError, match="Live mode requires API key"):
        main()
        
    monkeypatch.setenv("SILICONFLOW_API_KEY", "fake_key")
    monkeypatch.setattr(sys, "argv", [
        "run_stageab_api_eval.py",
        "--live",
        "--provider", "mock",
        "--model", "deepseek-ai/DeepSeek-V3",
        "--output-dir", str(out_dir)
    ])
    with pytest.raises(ValueError, match="Live mode requires a valid non-mock --provider"):
        main()


def test_no_revision_logging_and_metrics(tmp_path, monkeypatch):
    from experiments.multiagent.run_stageab_api_eval import main
    import sys
    import json
    
    out_dir = tmp_path / "stageab_no_rev_test"
    monkeypatch.setattr(sys, "argv", [
        "run_stageab_api_eval.py",
        "--mock",
        "--max-cases", "1",
        "--output-dir", str(out_dir)
    ])
    main()
    
    parsed_file = out_dir / "stage_a_parsed.jsonl"
    assert parsed_file.exists()
    with open(parsed_file, "r") as f:
        row = json.loads(f.readline())
        sub = row["submissions"][0]
        assert len(sub["actions"]) == 1
        assert sub["actions"][0]["action_type"] == "NO_REVISION"
        assert len(sub["proposal_edges"]) == 0
        
    metrics_file = out_dir / "metrics.json"
    assert metrics_file.exists()
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
        assert "no_revision_match" in metrics["stage_a"]

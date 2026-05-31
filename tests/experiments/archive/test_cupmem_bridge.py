from __future__ import annotations

import json
import pytest
from retracemem.multiagent.commit import commit_subagent_submission
from experiments.archive.cupmem_bridge import map_delta_to_submission, map_invalidation_to_submission
from experiments.archive.stale_cupmem_comparison import run_arm_a_original, run_arm_b_retrace, execute_offline_fixture_comparison
from experiments.archive.fixtures import get_diagnostic_fixtures
from experiments.archive.legacy.run_offline_diagnostic import check_trace_completeness


def test_e1_trace_completeness_validator():
    # 1. Valid trace structure
    mock_auth_trace = {
        "fine_grained_statuses": {"b_1": "BLOCKED"},
        "defeat_paths": [{"belief_id": "b_1", "path_type": "PREREQUISITE_BLOCK"}],
    }
    valid_commit_trace = {
        "producer_id": "agent_1",
        "producer_role": "role",
        "submission_id": "sub_1",
        "observed_at": "2026-05-29T10:00:00Z",
        "next_snapshot_id": "snap_1",
        "view_fingerprint": "fingerprint_1",
        "auth_trace": mock_auth_trace,
    }
    all_traces = {"scenario_1": [valid_commit_trace]}
    assert check_trace_completeness(all_traces) is True

    # 2. Invalid trace: missing defeat path for BLOCKED status
    invalid_auth_trace = {
        "fine_grained_statuses": {"b_1": "BLOCKED"},
        "defeat_paths": [],
    }
    invalid_commit_trace = dict(valid_commit_trace, auth_trace=invalid_auth_trace)
    assert check_trace_completeness({"scenario_1": [invalid_commit_trace]}) is False

    # 3. Invalid trace: missing provenance
    incomplete_commit_trace = dict(valid_commit_trace)
    del incomplete_commit_trace["producer_id"]
    assert check_trace_completeness({"scenario_1": [incomplete_commit_trace]}) is False


def test_cupmem_bridge_leakage_and_mapping():
    mock_delta = {
        "delta_id": "d_001",
        "session_id": "s_000",
        "session_index": 0,
        "session_time": "2026-05-29T10:00:00Z",
        "bucket": "workplace",
        "local_track": "current_employer",
        "proposed_value": "User works in Portland.",
        "source_type": "extraction",
        "evidence_chunk_ids": ["c_001"],
        "confidence": 0.9,
    }
    mock_chunks = [{"chunk_id": "c_001", "text": "User works in Portland."}]

    # Rejection of gold field leakage
    leaky_delta = dict(mock_delta, M_old="gold stuff")
    with pytest.raises(ValueError, match="Evaluation leakage detected"):
        map_delta_to_submission(leaky_delta, mock_chunks, [], "snap_root")

    # Correct conversion to SubagentMemorySubmission
    sub = map_delta_to_submission(mock_delta, mock_chunks, [], "snap_root")
    assert sub.submission_id == "d_001"
    assert sub.new_evidence_id == "c_001"
    assert len(sub.evidence_context) == 1
    assert sub.evidence_context[0].text == "User works in Portland."

    # Execution through commit_subagent_submission
    res = commit_subagent_submission(sub)
    assert res.submission_id == "d_001"
    assert "producer_id" in res.commit_trace


def test_stale_comparison_harness_mode_1():
    result = execute_offline_fixture_comparison()
    assert "manifest" in result
    assert "arm_a_result" in result
    assert "arm_b_result" in result
    
    # Ensure correct schema is exported
    active_b = result["arm_b_result"]["active_items"]
    assert len(active_b) == 1
    assert active_b[0]["value"] == "User works in Vancouver."


def test_e1_deterministic_replay():
    fixtures = get_diagnostic_fixtures()
    for name, submissions in fixtures.items():
        res1 = []
        res2 = []
        for sub in submissions:
            res1.append(commit_subagent_submission(sub).commit_trace)
            res2.append(commit_subagent_submission(sub).commit_trace)
        assert res1 == res2

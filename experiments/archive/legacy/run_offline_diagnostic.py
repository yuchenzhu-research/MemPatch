from __future__ import annotations

import json
import os
from typing import Any
from retracemem.multiagent.commit import commit_subagent_submission
from experiments.archive.fixtures import get_diagnostic_fixtures
from retracemem.schemas import AuthorizationStatus


def run_naive_baseline(scenario_name: str, submissions: tuple[Any, ...]) -> dict[str, str]:
    """A naive last-write-wins (LWW) baseline that ignores TM/DPA graph semantics.
    
    It blindly authorizes candidate beliefs and overwrites any existing state
    whenever new evidence is presented, ignoring blocks, releases, and uncertain qualifiers.
    """
    active_statuses: dict[str, str] = {}
    for sub in submissions:
        # Naive approach:
        # 1. Authorize all candidate beliefs in the submission
        for b in sub.candidate_beliefs:
            active_statuses[b.belief_id] = "AUTHORIZED"
        # 2. If replacement beliefs are specified, supersede old ones
        for b in sub.candidate_replacement_beliefs:
            active_statuses[b.belief_id] = "AUTHORIZED"
            # Attempt to mark the overwritten target as SUPERSEDES
            for target_id in active_statuses:
                if target_id != b.belief_id and "location" in target_id:
                    active_statuses[target_id] = "SUPERSEDED"
    return active_statuses


def evaluate_scenario(
    name: str,
    submissions: tuple[Any, ...],
) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
    # ReTrace evaluation
    retrace_statuses: dict[str, str] = {}
    traces = []
    for sub in submissions:
        res = commit_subagent_submission(sub)
        traces.append(res.commit_trace)
        
        # Accumulate/update statuses
        fine_grained = res.authorization_result.trace["fine_grained_statuses"]
        for bid, status in fine_grained.items():
            retrace_statuses[bid] = status

    # Naive baseline evaluation
    naive_statuses = run_naive_baseline(name, submissions)

    return retrace_statuses, naive_statuses, traces


def check_trace_completeness(all_traces: dict[str, list[dict[str, Any]]]) -> bool:
    for name, traces in all_traces.items():
        for commit_trace in traces:
            # 1. Check producer/submission provenance present in commit trace
            required_prov = {"producer_id", "producer_role", "submission_id", "observed_at", "next_snapshot_id"}
            if not all(k in commit_trace for k in required_prov):
                return False
            # 2. Check view_fingerprint present
            if "view_fingerprint" not in commit_trace:
                return False
            # 3. Check auth_trace / fine_grained_statuses / defeat path when blocked/superseded
            auth_trace = commit_trace.get("auth_trace", {})
            fine_grained = auth_trace.get("fine_grained_statuses", {})
            if not fine_grained:
                return False
            defeat_paths = auth_trace.get("defeat_paths", [])
            for bid, status in fine_grained.items():
                if status in ("BLOCKED", "SUPERSEDED"):
                    # Check if any defeat path targets this belief
                    has_path = any(p.get("belief_id") == bid for p in defeat_paths)
                    if not has_path:
                        return False
    return True


def main() -> None:
    fixtures = get_diagnostic_fixtures()
    results: dict[str, Any] = {}

    expected_retrace: dict[str, dict[str, str]] = {
        "conflict": {"b_vancouver": "AUTHORIZED", "b_portland": "SUPERSEDED", "b_location": "SUPERSEDED"},
        "stale_propagation": {"b_portland": "AUTHORIZED", "b_location": "SUPERSEDED"},
        "protected_belief": {"b_hobby": "AUTHORIZED"},
        "temporary_blocker": {"b_commute": "AUTHORIZED"},  # recovered in round 2
        "duplicate_evidence": {"b_dog": "AUTHORIZED"},
        "uncertain_update": {"b_cat": "UNRESOLVED"},
    }

    expected_naive: dict[str, dict[str, str]] = {
        # Naive will fail to block/supersede properly under complex scenarios
        "stale_propagation": {"b_location": "AUTHORIZED"},  # stale reaffirmation overwrites!
        "temporary_blocker": {"b_commute": "AUTHORIZED"},  # naive never blocks
        "uncertain_update": {"b_cat": "AUTHORIZED"},  # naive ignores uncertain
    }

    correct_retrace = 0
    total_retrace = 0
    protected_preservations = []
    unsupported_overwrite_count = 0
    replay_consistent = True

    all_scenario_traces = {}

    for name, submissions in fixtures.items():
        # Evaluate ReTrace and Naive
        retrace_res, naive_res, traces = evaluate_scenario(name, submissions)
        all_scenario_traces[name] = traces

        # Deterministic Replay Consistency check
        retrace_res_2, _, _ = evaluate_scenario(name, submissions)
        if retrace_res != retrace_res_2:
            replay_consistent = False

        # Status Accuracy Metric
        expected = expected_retrace[name]
        for bid, exp_status in expected.items():
            total_retrace += 1
            if retrace_res.get(bid) == exp_status:
                correct_retrace += 1

        # Protected-belief preservation (Scenario 3)
        if name == "protected_belief":
            protected_preservations.append(1.0 if retrace_res.get("b_hobby") == "AUTHORIZED" else 0.0)

        # Unsupported Overwrite Count (violations in baseline/stale/blocker)
        # 1. Stale propagation: b_location should NOT be AUTHORIZED in ReTrace
        if name == "stale_propagation" and retrace_res.get("b_location") == "AUTHORIZED":
            unsupported_overwrite_count += 1
        # 2. Temporary blocker round 1: b_commute should be BLOCKED
        if name == "temporary_blocker":
            # Let's inspect the intermediate state of the blocker scenario
            first_commit = commit_subagent_submission(submissions[0])
            first_commute_status = first_commit.authorization_result.trace["fine_grained_statuses"].get("b_commute")
            if first_commute_status != "BLOCKED":
                unsupported_overwrite_count += 1

    # Recovery Correctness
    recovery_correctness = 1.0
    temp_blocker_subs = fixtures.get("temporary_blocker")
    if temp_blocker_subs:
        retrace_res, _, _ = evaluate_scenario("temporary_blocker", temp_blocker_subs)
        if retrace_res.get("b_commute") == "AUTHORIZED":
            recovery_correctness = 1.0
        else:
            recovery_correctness = 0.0

    trace_completeness = check_trace_completeness(all_scenario_traces)

    # Summarize Metrics
    metrics = {
        "authorization_status_accuracy": correct_retrace / total_retrace if total_retrace > 0 else 0.0,
        "protected_belief_preservation": sum(protected_preservations) / len(protected_preservations) if protected_preservations else 1.0,
        "unsupported_overwrite_count": unsupported_overwrite_count,
        "recovery_correctness": recovery_correctness,
        "trace_completeness": trace_completeness,
        "deterministic_replay_consistency": replay_consistent,
    }


    output_data = {
        "metrics": metrics,
        "traces": all_scenario_traces,
    }

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/multiagent_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Metrics written to {out_path}:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

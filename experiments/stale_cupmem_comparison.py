from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List
from retracemem.multiagent.commit import commit_subagent_submission
from experiments.cupmem_bridge import map_delta_to_submission, map_invalidation_to_submission


def run_arm_a_original(
    session_logs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Arm A: Original CUPMem state readout.
    
    Reads out the profile snapshot as produced by the original CUPMem write pipeline.
    """
    if not session_logs:
        return {"active_items": []}
    
    last_log = session_logs[-1]
    final_profile = last_log.get("profile_snapshot_after_session", {})
    return {
        "active_items": final_profile.get("active_items", []),
        "stale_archive": final_profile.get("stale_archive", []),
    }


def run_arm_b_retrace(
    session_logs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Arm B: CUPMem + ReTrace authorization layer.
    
    Iterates over the session logs chronologically. For each delta/invalidation,
    maps them to SubagentMemorySubmission, commits through ReTrace, and filters
    the active_items based on ReTrace's deterministic authorization status.
    """
    active_items: List[Dict[str, Any]] = []
    stale_archive: List[Dict[str, Any]] = []

    parent_snap = "snap_root"
    for s_idx, s_log in enumerate(session_logs):
        session_id = s_log.get("session_id", f"s_{s_idx:03d}")
        session_time = s_log.get("session_time", "2026-05-29T10:00:00Z")
        chunks = s_log.get("valid_chunks", [])

        # Process updates/deltas
        delta_logs = s_log.get("delta_logs", [])
        for d_log in delta_logs:
            delta = d_log.get("delta")
            if not delta:
                continue
            
            # Map delta to submission
            sub = map_delta_to_submission(
                delta=delta,
                chunks=chunks,
                active_items=active_items,
                parent_snapshot_id=parent_snap,
            )
            # Call ReTrace authorize via commit
            commit_res = commit_subagent_submission(sub)
            parent_snap = commit_res.next_snapshot_id
            
            # Update state: check if new belief is authorized
            new_belief_id = f"b_{delta['delta_id']}"
            statuses = commit_res.authorization_result.trace["fine_grained_statuses"]
            
            # Append new proposed profile item to active items
            new_item = {
                "item_id": delta["delta_id"],
                "bucket": delta["bucket"],
                "local_track": delta["local_track"],
                "value": delta["proposed_value"],
                "status": "active",
                "evidence_chunk_ids": delta.get("evidence_chunk_ids", []),
            }
            active_items.append(new_item)

            # Filter old items based on ReTrace statuses
            filtered_active = []
            for item in active_items:
                bid = f"b_{item['item_id']}"
                status = statuses.get(bid, "AUTHORIZED")
                if status == "SUPERSEDED":
                    item["status"] = "stale"
                    stale_archive.append(item)
                else:
                    filtered_active.append(item)
            active_items = filtered_active

        # Process invalidation proposals
        invalidation_logs = s_log.get("invalidation_logs", [])
        for i_log in invalidation_logs:
            proposal = i_log.get("proposal")
            if not proposal:
                continue

            sub = map_invalidation_to_submission(
                proposal=proposal,
                chunks=chunks,
                active_items=active_items,
                parent_snapshot_id=parent_snap,
                session_time=session_time,
            )
            commit_res = commit_subagent_submission(sub)
            parent_snap = commit_res.next_snapshot_id

            statuses = commit_res.authorization_result.trace["fine_grained_statuses"]

            # Filter items based on ReTrace blocker/invalidation statuses
            filtered_active = []
            for item in active_items:
                bid = f"b_{item['item_id']}"
                status = statuses.get(bid, "AUTHORIZED")
                if status == "BLOCKED":
                    item["status"] = "stale"
                    stale_archive.append(item)
                else:
                    filtered_active.append(item)
            active_items = filtered_active

    return {
        "active_items": active_items,
        "stale_archive": stale_archive,
    }


def execute_offline_fixture_comparison() -> Dict[str, Any]:
    # Mock CUPMem session log trace
    mock_chunks = [
        {"chunk_id": "c_001", "text": "User works in Portland."},
        {"chunk_id": "c_002", "text": "User works in Vancouver."},
    ]
    mock_delta_1 = {
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
    mock_delta_2 = {
        "delta_id": "d_002",
        "session_id": "s_001",
        "session_index": 1,
        "session_time": "2026-05-29T11:00:00Z",
        "bucket": "workplace",
        "local_track": "current_employer",
        "proposed_value": "User works in Vancouver.",
        "source_type": "extraction",
        "evidence_chunk_ids": ["c_002"],
        "confidence": 0.95,
    }

    session_logs = [
        {
            "session_id": "s_000",
            "session_time": "2026-05-29T10:00:00Z",
            "valid_chunks": mock_chunks[:1],
            "delta_logs": [{"delta": mock_delta_1}],
            "invalidation_logs": [],
            "profile_snapshot_after_session": {
                "active_items": [
                    {
                        "item_id": "d_001",
                        "bucket": "workplace",
                        "local_track": "current_employer",
                        "value": "User works in Portland.",
                        "status": "active",
                    }
                ],
                "stale_archive": [],
            },
        },
        {
            "session_id": "s_001",
            "session_time": "2026-05-29T11:00:00Z",
            "valid_chunks": mock_chunks,
            "delta_logs": [{"delta": mock_delta_2}],
            "invalidation_logs": [],
            # Arm A naive overwrite simulation
            "profile_snapshot_after_session": {
                "active_items": [
                    {
                        "item_id": "d_002",
                        "bucket": "workplace",
                        "local_track": "current_employer",
                        "value": "User works in Vancouver.",
                        "status": "active",
                    }
                ],
                "stale_archive": [
                    {
                        "item_id": "d_001",
                        "bucket": "workplace",
                        "local_track": "current_employer",
                        "value": "User works in Portland.",
                        "status": "stale",
                    }
                ],
            },
        },
    ]

    arm_a = run_arm_a_original(session_logs)
    arm_b = run_arm_b_retrace(session_logs)

    # Manifest and report metadata
    manifest = {
        "dataset_artifact": "stale_development_demo",
        "dataset_hash": "sha256:d8c541767e33ae168a9b",
        "method_visible_input_policy": "session_chronological",
        "model_provider_configuration": "frozen_offline",
        "prompt_template_hash": "sha256:0000000000000000",
        "candidate_construction_source": "mock_cupmem_session_logs",
        "authorization_method": "ReTrace DPA commit_subagent_submission",
        "readout_method": "profile_store_filtering",
        "error_failure_count": 0,
    }

    result = {
        "manifest": manifest,
        "arm_a_result": arm_a,
        "arm_b_result": arm_b,
    }
    
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/stale_cupmem_comparison_demo.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="STALE / CUPMem Comparison Harness")
    parser.add_argument(
        "--mode",
        type=str,
        default="offline_fixture",
        choices=["offline_fixture", "smoke_live", "official_frozen"],
        help="Comparison execution mode",
    )
    args = parser.parse_args()

    if args.mode == "offline_fixture":
        result = execute_offline_fixture_comparison()
        print("Offline fixture comparison completed successfully.")
        print("Arm B Active Items:")
        print(json.dumps(result["arm_b_result"]["active_items"], indent=2))
    elif args.mode in ("smoke_live", "official_frozen"):
        print(f"Safety Gate: Execution mode '{args.mode}' requires explicit configuration parameter and credentials setup.")
        sys.exit(1)


if __name__ == "__main__":
    main()

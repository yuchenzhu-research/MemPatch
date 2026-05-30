#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any, Dict, List

def validate_plot_inputs(
    results_path: str,
    details_path: str | None = None,
    official: bool = False
) -> None:
    print(f"[*] Validating plot inputs from results: {results_path}")
    if not os.path.exists(results_path):
        print(f"[Error] Results file not found: {results_path}")
        sys.exit(1)

    # 1. Parse JSONL records
    records: List[Dict[str, Any]] = []
    with open(results_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[Error] JSON Decode error at line {idx}: {e}")
                sys.exit(1)

    if not records:
        print("[Error] No records found in results file.")
        sys.exit(1)

    print(f"[*] Loaded {len(records)} metric rows. Starting field validation...")

    # Define required fields per figure
    fig3_fields = ["method_name", "metric_name", "metric_value", "protocol_mode", "scientific_status"]
    fig4_fields = ["method_name", "number_of_subagents", "conflict_density", "delay_depth", "metric_name", "metric_value"]
    fig5_fields = ["method_name", "failure_type", "metric_name", "metric_value"]
    fig6_fields = ["method_name", "backbone_model", "metric_name", "metric_value", "calls", "tokens", "latency_ms"]

    for idx, r in enumerate(records, 1):
        # General structure: Check 23 required fields from Packet 4B Section 6
        required_general = [
            "run_id", "episode_id", "domain", "failure_type", "protocol_mode",
            "scientific_status", "split", "method_name", "backbone_model", "proposal_source",
            "candidate_source", "number_of_subagents", "number_of_submissions", "role_diversity",
            "conflict_density", "delay_depth", "recovery_present", "metric_name", "metric_value",
            "trace_available", "calls", "tokens", "latency_ms",
            "policy_variant", "checkpoint_id", "training_split", "training_step",
            "training_examples_seen", "reward_variant", "authorization_reward",
            "downstream_task_reward", "scope_expansion_penalty", "stale_penalty", "total_reward"
        ]
        for fld in required_general:
            if fld not in r:
                print(f"[Error] Row {idx} is missing required general field: '{fld}'")
                sys.exit(1)

        # Figure 3 Validation
        for fld in fig3_fields:
            if r.get(fld) is None and fld not in ["backbone_model"]: # None values allowed for backbone model, others must exist
                if fld == "scientific_status" and r.get(fld) is None:
                    print(f"[Error] Row {idx} has null value for Fig3 field '{fld}'")
                    sys.exit(1)

        # Figure 4 Validation
        for fld in fig4_fields:
            if r.get(fld) is None:
                print(f"[Error] Row {idx} has null value for Fig4 field '{fld}'")
                sys.exit(1)

        # Figure 5 Validation
        for fld in fig5_fields:
            if r.get(fld) is None:
                print(f"[Error] Row {idx} has null value for Fig5 field '{fld}'")
                sys.exit(1)

        # Figure 6 Validation (allow backbone_model, calls, tokens, latency_ms to be None/0 in offline replay)
        for fld in fig6_fields:
            if fld not in r:
                print(f"[Error] Row {idx} is missing Fig6 field '{fld}'")
                sys.exit(1)

        # 2. Strict Scientific Status check
        # "The validation script should fail if official/paper-ready plotting is attempted using only records tagged:
        #  scientific_status = 'mechanism_validation_only'"
        if official:
            sci_status = r.get("scientific_status")
            if sci_status in ("mechanism_validation_only", "pipeline_validation_only", "not_evaluated", "smoke_validation_only"):
                print(f"[Error] Official plotting rejected: Row {idx} is tagged with scientific_status '{sci_status}'")
                sys.exit(1)
            if r.get("split") in ("development_candidate", "unreviewed_candidate"):
                print(f"[Error] Official plotting rejected: Row {idx} is from unreviewed development_candidate split.")
                sys.exit(1)

    print("[+] All metric rows successfully validated for Figures 3, 4, 5, 6.")

    # 3. Validate qualitative timeline trace details if provided
    if details_path:
        print(f"[*] Validating qualitative trace timeline from details: {details_path}")
        if not os.path.exists(details_path):
            print(f"[Error] Details file not found: {details_path}")
            sys.exit(1)
        try:
            with open(details_path, "r", encoding="utf-8") as f:
                details_data = json.load(f)
        except Exception as e:
            print(f"[Error] Failed to parse details JSON: {e}")
            sys.exit(1)

        if not isinstance(details_data, list):
            print("[Error] Details JSON must be a list of episode results.")
            sys.exit(1)

        for ep_res in details_data:
            # Check outer fields
            if "episode_id" not in ep_res:
                print("[Error] Episode result detail missing 'episode_id'")
                sys.exit(1)
            
            # If it contains revision_events (E0 Replay)
            if "revision_events" in ep_res:
                events = ep_res["revision_events"]
                for ev in events:
                    required_event_fields = [
                        "submission_id", "producer_role", "parent_snapshot_id",
                        "next_snapshot_id", "belief_statuses", "trace_available"
                    ]
                    for fld in required_event_fields:
                        if fld not in ev:
                            print(f"[Error] Revision event in episode {ep_res.get('episode_id')} missing field: '{fld}'")
                            sys.exit(1)
            # If it is Fixed-Candidate (E1 Replay) which uses decisions
            elif "decisions" in ep_res:
                decisions = ep_res["decisions"]
                for dec in decisions:
                    required_dec_fields = ["belief_id", "decision", "trace_available"]
                    # check belief_id and decision
                    if "belief_id" not in dec or "decision" not in dec:
                        print(f"[Error] Decision in episode {ep_res.get('episode_id')} missing core fields")
                        sys.exit(1)
        
        print("[+] Qualitative trace details validated successfully.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate plot input datasets for ReTrace figures.")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSONL file")
    parser.add_argument("--details", type=str, default=None, help="Path to details JSON file for qualitative timeline")
    parser.add_argument("--official", action="store_true", help="Perform strict check for paper-ready official plotting")
    args = parser.parse_args()

    try:
        validate_plot_inputs(args.results, args.details, args.official)
        print("[+] DATASET IS PLOT-READY AND STRUCTURALLY COMPLETE.")
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        print(f"[Error] Unexpected validation error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

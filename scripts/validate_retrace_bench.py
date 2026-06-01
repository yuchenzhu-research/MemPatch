#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

# Add root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.retrace_bench.schemas import Scenario, DialogueTurn, MemoryEntry, ProbeQuery
from benchmark.retrace_bench.taxonomy import Domain, RevisionFamily, FinalStatus, ProbeType, RevisionActionType
from benchmark.retrace_bench.generation.validate_generated import validate_scenarios
from benchmark.retrace_bench.utils.jsonl import read_jsonl


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate ReTrace-Bench dataset.")
    parser.add_argument("--data", required=True, help="Path to data directory containing scenarios.jsonl")
    args = parser.parse_args(argv)

    data_dir = Path(args.data)
    scenarios_path = data_dir / "scenarios.jsonl"
    manifest_path = data_dir / "manifest.json"

    if not scenarios_path.exists():
        print(f"Error: {scenarios_path} does not exist.")
        return 1

    if not manifest_path.exists():
        print(f"Error: {manifest_path} does not exist.")
        return 1

    # Load and check Manifest
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    
    if not manifest_data.get("metadata", {}).get("evaluation_only"):
        print("Error: Manifest must explicitly mark evaluation_only as True.")
        return 1

    print(f"Loading scenarios from {scenarios_path}...")
    records = read_jsonl(scenarios_path)
    
    # Parse back into Scenario objects
    scenarios = []
    for r in records:
        dialogue = [
            DialogueTurn(speaker=t["speaker"], text=t["text"], timestamp=t.get("timestamp"), metadata=t.get("metadata", {}))
            for t in r.get("dialogue_history", [])
        ]
        snapshot = [
            MemoryEntry(entry_id=e["entry_id"], content=e["content"], entry_type=e["entry_type"], metadata=e.get("metadata", {}))
            for e in r.get("memory_snapshot", [])
        ]
        from benchmark.retrace_bench.schemas import RevisionAction
        actions = [
            RevisionAction(
                action_type=RevisionActionType(a["action_type"]),
                target_id=a["target_id"],
                replacement_id=a.get("replacement_id"),
                evidence_ids=a.get("evidence_ids", []),
                rationale=a.get("rationale")
            )
            for a in r.get("gold_revision_actions", [])
        ]
        queries = [
            ProbeQuery(
                query_id=q["query_id"],
                probe_type=ProbeType(q["probe_type"]),
                question=q["question"],
                options=q["options"],
                gold_answer=q["gold_answer"]
            )
            for q in r.get("probe_queries", [])
        ]
        scenarios.append(Scenario(
            scenario_id=r["scenario_id"],
            domain=Domain(r["domain"]),
            revision_family=RevisionFamily(r["revision_family"]),
            conflict_type=r["conflict_type"],
            memory_topology=r["memory_topology"],
            dialogue_history=dialogue,
            memory_snapshot=snapshot,
            gold_final_statuses={k: FinalStatus(v) for k, v in r["gold_final_statuses"].items()},
            gold_revision_actions=actions,
            probe_queries=queries,
            metadata=r.get("metadata", {})
        ))

    print(f"Validating {len(scenarios)} scenarios...")
    report, accepted, rejected = validate_scenarios(scenarios)

    if not report.is_valid:
        print("Validation FAILED!")
        for error in report.errors:
            print(f"  - {error}")
        return 1

    print("Validation SUCCESSFUL!")
    print(f"  Total Checked: {report.num_checked}")
    print(f"  Evaluation-Only Flag Checked: YES")
    return 0


if __name__ == "__main__":
    sys.exit(main())

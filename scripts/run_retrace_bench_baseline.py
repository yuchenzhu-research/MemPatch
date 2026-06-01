#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmark.retrace_bench.schemas import Scenario, DialogueTurn, MemoryEntry, ProbeQuery
from benchmark.retrace_bench.taxonomy import Domain, RevisionFamily, FinalStatus, ProbeType, RevisionActionType
from benchmark.retrace_bench.utils.jsonl import read_jsonl, write_jsonl
from benchmark.retrace_bench.llm_providers import get_provider
from benchmark.retrace_bench.evaluation.run_evaluation import run_evaluation_loop


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run ReTrace-Bench baseline evaluation.")
    parser.add_argument("--data", required=True, help="Path to data directory containing scenarios.jsonl")
    parser.add_argument("--baseline", required=True, help="Name of baseline model to run")
    parser.add_argument("--out", required=True, help="Path to output predictions.jsonl")
    parser.add_argument("--provider", default=None, help="LLM provider name (optional)")
    parser.add_argument("--model", default=None, help="LLM model name (optional)")
    parser.add_argument("--api-key", default=None, help="LLM API Key (optional)")
    args = parser.parse_args(argv)

    scenarios_path = Path(args.data) / "scenarios.jsonl"
    if not scenarios_path.exists():
        print(f"Error: {scenarios_path} does not exist.")
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

    # Initialize provider if requested
    provider = None
    if args.provider and args.model:
        print(f"Initializing provider {args.provider} with model {args.model}...")
        provider = get_provider(args.provider, args.model, api_key=args.api_key)

    print(f"Running baseline '{args.baseline}'...")
    predictions = run_evaluation_loop(scenarios, args.baseline, provider=provider)

    print(f"Saving predictions to {args.out}...")
    write_jsonl(args.out, predictions)
    print("Baseline run completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

# Add root to python path to allow imports when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.retrace_bench.generation.expand_scenarios import expand_templates, DOMAINS
from benchmark.retrace_bench.generation.validate_generated import validate_scenarios
from benchmark.retrace_bench.utils.jsonl import write_jsonl
from benchmark.retrace_bench.utils.splits import split_list


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build ReTrace-Bench dataset.")
    parser.add_argument("--out", required=True, help="Output directory path")
    parser.add_argument("--num-scenarios", type=int, default=100, help="Number of scenarios to generate")
    parser.add_argument("--queries-per-scenario", type=int, default=4, help="Queries per scenario (must be 4)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for generation")
    args = parser.parse_args(argv)

    if args.queries_per_scenario != 4:
        print("Error: --queries-per-scenario must be exactly 4.")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_scenarios} scenarios with seed {args.seed}...")
    scenarios = expand_templates(args.num_scenarios, seed=args.seed)

    print("Validating generated scenarios...")
    report, accepted, rejected = validate_scenarios(scenarios)

    # Write rejected examples if any
    rejected_path = out_dir / "rejected_examples.jsonl"
    write_jsonl(rejected_path, rejected)
    if rejected:
        print(f"Warning: {len(rejected)} scenarios failed validation. Details in {rejected_path}")

    # Write scenarios and all queries
    write_jsonl(out_dir / "scenarios.jsonl", accepted)
    
    all_queries = []
    audit_queries = []
    for s in accepted:
        all_queries.extend(s.probe_queries)
        for q in s.probe_queries:
            if q.probe_type.value == "audit_localization":
                audit_queries.append(q)
    
    write_jsonl(out_dir / "queries.jsonl", all_queries)
    write_jsonl(out_dir / "audit_sample.jsonl", audit_queries)

    # Deterministic split
    dev_scens, test_scens, private_scens = split_list(accepted, dev_ratio=0.2, test_ratio=0.7, seed=args.seed)

    write_jsonl(out_dir / "public_dev.jsonl", dev_scens)
    write_jsonl(out_dir / "public_test.jsonl", test_scens)
    write_jsonl(out_dir / "private_test_stub.jsonl", private_scens)

    # Write Manifest
    manifest_data = {
        "version": "1.0.0",
        "num_scenarios": len(accepted),
        "num_queries": len(all_queries),
        "domains": [d.value for d in DOMAINS],
        "created_at": "2026-06-01T10:43:08+08:00",
        "metadata": {
            "evaluation_only": True,
            "seed": args.seed,
            "description": "ReTrace-Bench evaluation-only dataset"
        }
    }
    
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated dataset at {out_dir}:")
    print(f"  Accepted scenarios: {len(accepted)}")
    print(f"  Total queries: {len(all_queries)}")
    print(f"  Dev split: {len(dev_scens)} scenarios")
    print(f"  Test split: {len(test_scens)} scenarios")
    print(f"  Private split: {len(private_scens)} scenarios")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.retrace_bench.evaluation.aggregate_results import aggregate_and_save


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate ReTrace-Bench results.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    parser.add_argument("--scenarios", default=None, help="Path to scenarios.jsonl (optional, default to guessing based on smoke)")
    parser.add_argument("--out", required=True, help="Path to output aggregate report.json")
    args = parser.parse_args(argv)

    predictions_path = Path(args.predictions)
    out_path = Path(args.out)

    # Resolve scenarios.jsonl path
    if args.scenarios:
        scenarios_path = Path(args.scenarios)
    else:
        # Default fallback guessing
        if "smoke" in str(predictions_path):
            scenarios_path = predictions_path.parent / "retrace_bench_v1" / "scenarios.jsonl"
            if not scenarios_path.exists():
                # Try outputs/smoke/retrace_bench_v1/scenarios.jsonl
                scenarios_path = Path("outputs/smoke/retrace_bench_v1/scenarios.jsonl")
            if not scenarios_path.exists():
                scenarios_path = Path("data/retrace_bench/test_800_templateheldout_en/scenarios.jsonl")
        else:
            scenarios_path = Path("data/retrace_bench/test_800_templateheldout_en/scenarios.jsonl")

    if not scenarios_path.exists():
        print(f"Error: scenarios.jsonl not found at {scenarios_path}. Please specify using --scenarios.")
        return 1

    if not predictions_path.exists():
        print(f"Error: predictions.jsonl not found at {predictions_path}.")
        return 1

    print(f"Aggregating results from predictions: {predictions_path}")
    print(f"Using scenarios from: {scenarios_path}")

    metrics = aggregate_and_save(scenarios_path, predictions_path, out_path)

    print(f"Aggregate report written to {out_path}:")
    print(f"  Overall query accuracy:        {metrics.get('overall_accuracy', 0.0):.2%}")
    print(f"  State Resolution accuracy:     {metrics.get('state_resolution_accuracy', 0.0):.2%}")
    print(f"  Premise Resistance accuracy:   {metrics.get('premise_resistance_accuracy', 0.0):.2%}")
    print(f"  Policy Adaptation accuracy:    {metrics.get('policy_adaptation_accuracy', 0.0):.2%}")
    print(f"  Audit Localization score:      {metrics.get('audit_localization_score', 0.0):.2%}")
    print(f"  Final Status match accuracy:   {metrics.get('final_status_accuracy', 0.0):.2%}")
    print(f"  Stale Propagation rate:        {metrics.get('stale_propagation_rate', 0.0):.2%}")
    print(f"  Over update rate:              {metrics.get('over_update_rate', 0.0):.2%}")
    print(f"  Under update rate:             {metrics.get('under_update_rate', 0.0):.2%}")
    print(f"  Total evaluated queries:       {metrics.get('total_evaluated_queries', 0)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

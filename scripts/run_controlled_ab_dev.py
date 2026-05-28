#!/usr/bin/env python3
"""AB-1B replay-only internal controlled A/B development runner.

WARNING: This is an INTERNAL DEVELOPMENT PROTOCOL CHECK ONLY.
- Replay/mock execution only.
- NOT an official benchmark.
- NOT strict call-budget matched.
- NO claim that ReTrace outperforms DirectJudge.

Usage:
    .venv/bin/python scripts/run_controlled_ab_dev.py [--output-dir outputs/controlled_ab_dev]

The runner:
1. Loads internal dev controlled cases from data/internal_dev/controlled_ab_cases.json.
2. Runs Stage A (ControlledReTraceLLM) and Stage B (DirectJudgeLLM) offline.
3. Computes controlled A/B metrics.
4. Emits JSON-compatible per-instance results and compact aggregate summary.
5. Writes output to an ignored/generated output location.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

# Ensure src is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from retracemem.evaluation.controlled_ab import (
    compute_metrics,
    format_report,
    load_cases,
    run_case,
)

_CASES_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "data", "internal_dev", "controlled_ab_cases.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AB-1B replay-only internal controlled A/B development runner."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), os.pardir, "outputs", "controlled_ab_dev"),
        help="Output directory for results (gitignored).",
    )
    parser.add_argument(
        "--cases",
        default=_CASES_PATH,
        help="Path to internal dev cases JSON file.",
    )
    args = parser.parse_args()

    cases_path = os.path.normpath(args.cases)
    output_dir = os.path.normpath(args.output_dir)

    print("=" * 70)
    print("AB-1B REPLAY-ONLY INTERNAL CONTROLLED A/B DEVELOPMENT RUNNER")
    print("=" * 70)
    print()
    print("  WARNING: Internal development protocol check only.")
    print("  Replay/mock execution only. NOT an official benchmark.")
    print("  NOT strict call-budget matched.")
    print("  NO claim that ReTrace outperforms DirectJudge.")
    print()
    print(f"  Cases: {cases_path}")
    print(f"  Output: {output_dir}")
    print()

    # Load cases
    cases = load_cases(cases_path)
    print(f"Loaded {len(cases)} internal development cases.")

    # Execute all cases
    results = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, case in enumerate(cases):
            print(f"  [{i+1}/{len(cases)}] {case.case_id} ({case.case_type})")
            res = run_case(case, tmp_dir)
            if res.stage_a_error:
                print(f"        Stage A ERROR: {res.stage_a_error}")
            if res.stage_b_error:
                print(f"        Stage B ERROR: {res.stage_b_error}")
            results.append(res)

    # Compute metrics
    metrics = compute_metrics(cases, results)

    # Format report
    report = format_report(metrics, results)

    # Print compact summary
    print()
    print("-" * 70)
    print("AGGREGATE SUMMARY")
    print("-" * 70)
    agg = report["aggregate"]
    print(f"  Total cases:              {agg['total_cases']}")
    print(f"  Total belief decisions:   {agg['total_belief_decisions']}")
    print(f"  Stage A accuracy:         {agg['stage_a_accuracy']}")
    print(f"  Stage B accuracy:         {agg['stage_b_accuracy']}")
    print(f"  Stage A status breakdown: {agg['stage_a_status_breakdown']}")
    print(f"  Stage B verdict breakdown:{agg['stage_b_verdict_breakdown']}")
    print(f"  Obsolete misuse (A):      {agg['obsolete_misuse']}")
    print(f"  Protected preserved (A):  {agg['protected_belief_preserved']}")
    print(f"  Rollback recovery (A):    {agg['rollback_recovery']}")
    print(f"  Unsupported revision:     {agg['unsupported_revision_rate']}")
    print(f"  Execution errors:         {agg['execution_errors']}")
    print()
    print("  Observed cost (NOT matched):")
    for stage, label in [("stage_a", "Stage A"), ("stage_b", "Stage B")]:
        c = agg["observed_cost"][stage]
        print(f"    {label}: calls={c['calls']}, tokens={c['tokens']}, "
              f"cache_hits={c['cache_hits']}, latency_ms={c['latency_ms']}")
    print()

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "controlled_ab_dev_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Full report written to: {output_path}")
    print()
    print("=" * 70)
    print("DONE. Internal development protocol check complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()

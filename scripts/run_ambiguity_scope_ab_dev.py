#!/usr/bin/env python3
"""Ambiguity-and-Scope Stage A/B feasibility runner.

Internal development diagnostic only.
- Not an official benchmark.
- Replay/mock mode is for runner correctness.
- Live mode is exploratory development-only via the validated provider/cache/accounting boundary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from retracemem.evaluation.ambiguity_scope import (
    compute_metrics,
    format_report,
    load_dataset,
    run_cases,
    validate_dataset_balance,
)
from retracemem.evaluation.manifest import (
    RunConfiguration,
    RunManifest,
    compute_file_sha256,
)

DEFAULT_DATASET = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "data",
    "internal_dev",
    "ambiguity_scope_controlled_v0.json",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, "outputs", "ambiguity_scope_dev"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage A/B Ambiguity-and-Scope feasibility runner (dev-only).",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Path to the internal Ambiguity-and-Scope dataset JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the report/manifest (gitignored).",
    )
    parser.add_argument(
        "--mode",
        choices=("replay", "live-dev"),
        default="replay",
        help="Execution mode. 'live-dev' is exploratory development only.",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        help="Live provider name (used only with --mode live-dev).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Live model id (used only with --mode live-dev).",
    )
    parser.add_argument(
        "--skip-balance-check",
        action="store_true",
        help="Skip the strict 4-per-category balance check (development bypass).",
    )
    args = parser.parse_args()

    dataset_path = os.path.normpath(args.dataset)
    output_dir = os.path.normpath(args.output_dir)
    run_id = f"run-ambiguity-scope-{uuid.uuid4()}"

    print("=" * 70)
    print("AMBIGUITY-AND-SCOPE STAGE A/B FEASIBILITY RUNNER")
    print("=" * 70)
    print("  Disclaimer: internal development feasibility study only.")
    print(f"  Run ID: {run_id}")
    print(f"  Mode:   {args.mode.upper()}")
    print(f"  Cases:  {dataset_path}")
    print(f"  Output: {output_dir}")
    print()

    cases = load_dataset(dataset_path)
    if not args.skip_balance_check:
        validate_dataset_balance(cases)
    print(f"Loaded {len(cases)} cases.")

    if args.mode == "live-dev":
        print("Refusing live execution in this runner until an explicit live-dev "
              "configuration is wired through CachedLLMClient.")
        print("Use replay mode for runner correctness and tests.")
        sys.exit(2)

    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        results = run_cases(cases, tmp)
        metrics = compute_metrics(cases, results)
        report = format_report(cases, results, metrics)

    report_path = Path(output_dir) / "ambiguity_scope_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {report_path}")

    config = RunConfiguration(
        run_id=run_id,
        stage_and_method_name="AmbiguityScope_StageAB_dev",
        provider_name="mock" if args.mode == "replay" else args.provider,
        model_id="mock" if args.mode == "replay" else args.model,
        cache_path="",
        dataset_checksum=compute_file_sha256(dataset_path),
        comparison_regime="ambiguity_scope_controlled",
    )
    manifest = RunManifest(
        config=config,
        aggregate_cost={
            "stage_a": {
                "calls": metrics.stage_a_calls,
                "tokens": metrics.stage_a_tokens,
            },
            "stage_b": {
                "calls": metrics.stage_b_calls,
                "tokens": metrics.stage_b_tokens,
            },
        },
        instance_count=len(cases),
        output_path=str(report_path),
    )
    manifest_path = Path(output_dir) / "ambiguity_scope_manifest.json"
    manifest.save(str(manifest_path))
    print(f"Manifest saved to: {manifest_path}")

    print()
    print("-" * 70)
    print("Summary")
    print("-" * 70)
    agg = report["aggregate"]
    print(f"  Overall accuracy A/B:               "
          f"{agg['overall_comparable_accuracy']['stage_a']} vs "
          f"{agg['overall_comparable_accuracy']['stage_b']}")
    print(f"  Stale blocking A/B:                 "
          f"{agg['stale_blocking_accuracy']['stage_a']} vs "
          f"{agg['stale_blocking_accuracy']['stage_b']}")
    print(f"  Protected preservation A/B:         "
          f"{agg['protected_belief_preservation']['stage_a']} vs "
          f"{agg['protected_belief_preservation']['stage_b']}")
    print(f"  Abstention accuracy A/B:            "
          f"{agg['abstention_accuracy']['stage_a']} vs "
          f"{agg['abstention_accuracy']['stage_b']}")
    print(f"  Unsupported confident revision A/B: "
          f"{agg['unsupported_confident_revision_rate']['stage_a']} vs "
          f"{agg['unsupported_confident_revision_rate']['stage_b']}")
    print(f"  Execution / parse errors:           "
          f"{agg['execution_errors']} / {agg['parse_errors']}")
    print()
    print("Reminder: this output is an internal feasibility diagnostic, not a benchmark result.")


if __name__ == "__main__":
    main()

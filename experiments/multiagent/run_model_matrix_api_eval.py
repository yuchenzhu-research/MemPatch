#!/usr/bin/env python3
"""
SiliconFlow / Industrial API Model Matrix Evaluation.
Compares DirectJudge-API, StageA-Freeform, StageA-Constrained, and StageC-ICL
across various SiliconFlow models (V3, Qwen) on dev_expansion.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import yaml
from pathlib import Path
from typing import Any

from experiments.multiagent.run_stageab_api_eval import run_stageab_eval, EvalRunConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="SiliconFlow Model Matrix Evaluation Runner")
    parser.add_argument("--config", default="configs/model_matrix.siliconflow.yaml", help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Dry run prompt checking and mockup metrics calculation")
    parser.add_argument("--max-cases", type=int, default=None, help="Force limit number of cases evaluated")
    parser.add_argument("--api-key", default=None, help="Explicit SILICONFLOW_API_KEY")
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    provider = config.get("provider", "siliconflow")
    api_url = config.get("api_url", "https://api.siliconflow.cn/v1/chat/completions")
    models = config.get("models", [])
    methods = config.get("methods", [])
    max_cases = args.max_cases or config.get("max_cases")

    print("=" * 80)
    print("SILICONFLOW INDUSTRIAL API MODEL MATRIX EVALUATOR (THIN WRAPPER)")
    print("=" * 80)
    print(f"Dry-run Mode: {args.dry_run}")
    print(f"Models: {models}")
    print(f"Methods: {methods}")
    print(f"Max Cases: {max_cases}")
    print("-" * 80)

    # API Setup
    api_key = args.api_key or os.getenv("SILICONFLOW_API_KEY")
    if not args.dry_run and not api_key:
        raise ValueError("SILICONFLOW_API_KEY is required for live API runs. Pass via --api-key or environment variable.")

    # Initialize Output Dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.get("output_dir", "outputs/runs/matrix_eval")) / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Matrix results structure: results[model][method] = { ...metrics }
    matrix_results: dict[str, dict[str, Any]] = {}

    effective_methods_seen = set()
    fallback_policy_allowed = config.get("allow_fallback_to_zeroshot", False) or args.dry_run

    for model in models:
        matrix_results[model] = {}
        for method in methods:
            requested_method = method
            effective_method = method

            allow_fallback = (requested_method == "StageC-ICL") and fallback_policy_allowed
            if requested_method == "StageC-ICL" and allow_fallback:
                effective_method = "zero_shot_fallback"

            effective_methods_seen.add(effective_method)

            if requested_method != effective_method:
                print(f"\nEvaluating Model: {model} | Method: {requested_method} (effective: {effective_method}) ...")
                print(f"  Requested={requested_method}, Effective={effective_method} (fallback_used=True)")
            else:
                print(f"\nEvaluating Model: {model} | Method: {requested_method} ...")

            # Determine proposer constraints
            constrained = (requested_method == "StageA-Constrained")

            # Prepare sub-config for run_stageab_eval
            # If dry_run is True, we run in mock replay mode (live=False, mock=True)
            # If dry_run is False, we run in live mode (live=True, mock=False)
            eval_config = EvalRunConfig(
                live=not args.dry_run,
                dry_run=False,
                mock=args.dry_run,
                max_cases=max_cases,
                resume=False,
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=api_url,
                output_dir=str(output_dir / f"{model.replace('/', '_')}_{effective_method}"),
                constrained=constrained,
                diagnostic=False,
                method=requested_method,
                allow_fallback_to_zeroshot=allow_fallback,
            )

            # Invoke authoritative eval engine
            global_metrics, _ = run_stageab_eval(eval_config)

            # Extract metrics based on method kind
            # DirectJudge-API corresponds to Stage B (direct judge) metrics
            # Stage A / Stage C methods correspond to Stage A (decomposition + DPA) metrics
            metric_source_key = "stage_b" if effective_method == "DirectJudge-API" else "stage_a"
            metrics = global_metrics[metric_source_key]

            # In DirectJudge, parser errors are tracked as the inverse of valid output rate
            # In Stage A, parser_error_rate is directly tracked
            if effective_method == "DirectJudge-API":
                parser_err_rate = 1.0 - metrics.get("valid_output_rate", 1.0)
            else:
                parser_err_rate = metrics.get("parser_error_rate", 0.0)

            # Map fields back to Matrix evaluation schema
            matrix_results[model][effective_method] = {
                "requested_method": requested_method,
                "effective_method": effective_method,
                "fallback_used": (requested_method != effective_method),
                "dpa_final_status_accuracy": metrics.get("final_status_accuracy", 0.0),
                "over_update_rate": metrics.get("over_update_rate", 0.0),
                "under_update_rate": metrics.get("under_update_rate", 0.0),
                "uncertainty_error_rate": metrics.get("uncertainty_error_rate", 0.0),
                "parser_error_rate": parser_err_rate,
                "total_beliefs_evaluated": metrics.get("total_beliefs", 0),
            }

            print(f"  Accuracy: {matrix_results[model][effective_method]['dpa_final_status_accuracy']:.4f}")
            print(f"  Over Updates (Stale Propagation): {matrix_results[model][effective_method]['over_update_rate']:.4f}")

    # Write Matrix results
    with open(output_dir / "matrix_results.json", "w", encoding="utf-8") as f:
        json.dump(matrix_results, f, indent=2)

    # Save manifest.json
    manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "run_identifier": "development_live_api_run / not_final_paper_result" if not args.dry_run else "development_run",
        "is_live_api_result": not args.dry_run,
        "mock_default_used": args.dry_run,
        "provider": provider,
        "models_evaluated": models,
        "requested_methods": methods,
        "effective_methods": sorted(list(effective_methods_seen)),
        "fallback_policy": "allow_fallback_to_zeroshot" if fallback_policy_allowed else "fail_closed",
        "output_directory": str(output_dir),
        "results": matrix_results,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Completed matrix evaluation successfully! Report saved to {output_dir}/")


if __name__ == "__main__":
    main()

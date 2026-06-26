#!/usr/bin/env python3
"""Internal MemPatch-Bench evaluator for raw hidden-gold scenario files.

For public/labels release bundles, prefer the label-based scorer:

    MemPatch score --labels release/labels/main_test_synthetic.labels.jsonl \
        --predictions path/to/predictions.jsonl --output scores.jsonl

This compatibility script still scores an existing JSONL predictions file
against raw internal scenarios that contain ``hidden_gold``.

Example::

    python scripts/evaluate_mempatch_predictions.py \
        --data local/data/mempatch/v1.4/raw_internal/main_test_synthetic.jsonl \
        --predictions path/to/predictions.jsonl \
        --out-metrics local/mempatch/my_model.metrics.json \
        --out-scored local/mempatch/my_model.scored.jsonl \
        --print-table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._root import bootstrap_from

bootstrap_from(__file__)

from mempatch.benchmark.api import (  # noqa: E402
    HEADLINE_METRICS,
    evaluate_predictions,
    load_predictions,
    load_scenarios,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official MemPatch-Bench evaluator for external predictions.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="scenarios.jsonl file or a directory containing scenarios.jsonl",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="JSONL predictions file (canonical or flat response format)",
    )
    parser.add_argument(
        "--out-metrics",
        default=None,
        help="Path to write aggregate metrics JSON",
    )
    parser.add_argument(
        "--out-scored",
        default=None,
        help="Optional path to write per-scenario scored predictions JSONL",
    )
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Raise on any validation error (default)",
    )
    strict_group.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Score what can be scored; report problems as warnings/errors",
    )
    parser.set_defaults(strict=True)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Convenience flag implying --no-strict (tolerate missing predictions)",
    )
    parser.add_argument(
        "--print-table",
        action="store_true",
        help="Print headline metrics to stdout",
    )
    return parser.parse_args(argv)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _print_table(result: dict[str, Any]) -> None:
    print(f"\nMemPatch-Bench headline metrics (scored {result['count']} predictions)")
    print("-" * 56)
    for key in HEADLINE_METRICS:
        value = result["headline_metrics"].get(key)
        if value is not None:
            print(f"  {key:<32} {value:.3f}")
    print("-" * 56)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    strict = args.strict and not args.allow_missing

    scenarios = load_scenarios(args.data)
    predictions = load_predictions(args.predictions)

    try:
        result = evaluate_predictions(
            scenarios,
            predictions,
            strict=strict,
            allow_missing=args.allow_missing,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for warning in result["warnings"]:
        if args.allow_missing and "missing prediction" in warning:
            continue
        print(f"warning: {warning}", file=sys.stderr)
    if not strict:
        for err in result["errors"]:
            if args.allow_missing and "missing prediction" in err:
                continue
            print(f"error (non-strict): {err}", file=sys.stderr)

    if args.out_metrics:
        out_metrics = Path(args.out_metrics)
        out_metrics.parent.mkdir(parents=True, exist_ok=True)
        metrics_payload = {
            "count": result["count"],
            "headline_metrics": result["headline_metrics"],
            "auxiliary_metrics": result["auxiliary_metrics"],
            "all_metrics": result["all_metrics"],
            "warnings": result["warnings"],
            "errors": result["errors"],
            "missing_prediction_count": result["missing_prediction_count"],
        }
        out_metrics.write_text(
            json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"Wrote metrics to {out_metrics}")

    if args.out_scored:
        _write_jsonl(Path(args.out_scored), result["scored_predictions"])
        print(f"Wrote scored predictions to {args.out_scored}")

    if args.print_table:
        _print_table(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

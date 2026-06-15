"""Fail-fast validation for smoke and completed model runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.api import evaluate_predictions, load_predictions, load_scenarios

try:
    from .run_core import ALL_METHODS
except ImportError:
    from run_core import ALL_METHODS


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-cases", type=int)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    raw_rows = _read_jsonl(run_dir / "raw_cases.jsonl")
    ids = [str(row["scenario_id"]) for row in raw_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("raw_cases.jsonl contains duplicate scenario IDs")
    if args.expected_cases is not None and len(raw_rows) != args.expected_cases:
        raise ValueError(f"expected {args.expected_cases} cases, found {len(raw_rows)}")

    for row in raw_rows:
        missing = set(ALL_METHODS) - set(row.get("predictions", {}))
        if missing:
            raise ValueError(f"{row['scenario_id']}: missing methods {sorted(missing)}")
        shared = row["generations"].get("mempatch_shared_actions") or {}
        if "actions_text" not in shared or "parse_result" not in shared:
            raise ValueError(f"{row['scenario_id']}: missing shared action record")
        guarded = row["predictions"]["mempatch"]
        unguarded = row["predictions"]["mempatch_no_guard"]
        if guarded["scenario_id"] != unguarded["scenario_id"]:
            raise ValueError(f"{row['scenario_id']}: paired variants are misaligned")

    scenarios = load_scenarios(args.data)
    selected_ids = set(ids)
    selected = [row for row in scenarios if str(row["scenario_id"]) in selected_ids]
    if len(selected) != len(raw_rows):
        raise ValueError("raw outputs contain IDs absent from the dataset")

    report = {"cases": len(raw_rows), "methods": {}}
    for method in ALL_METHODS:
        path = run_dir / f"{method}.predictions.jsonl"
        predictions = load_predictions(path)
        result = evaluate_predictions(selected, predictions, strict=False, allow_missing=False)
        if result["missing_prediction_count"]:
            raise ValueError(f"{method}: incomplete predictions")
        report["methods"][method] = {
            "errors": len(result["errors"]),
            "warnings": len(result["warnings"]),
            "headline_metrics": result["headline_metrics"],
        }

    target = run_dir / "validation_report.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

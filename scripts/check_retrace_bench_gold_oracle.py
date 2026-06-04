#!/usr/bin/env python3
"""Verify that hidden_gold replays score perfectly under the official scorer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.retrace_bench.general_taxonomy import canonical_hidden_gold_fields
from benchmark.retrace_bench.scorers_general import aggregate_metrics, score_prediction


GOLD_ORACLE_EXPECTATIONS = {
    "decision_macro_f1": 1.0,
    "black_box_decision_accuracy": 1.0,
    "memory_state_accuracy": 1.0,
    "evidence_f1": 1.0,
    "minimal_evidence_exact_match": 1.0,
    "failure_diagnosis_accuracy": 1.0,
    "stale_reuse_rate": 0.0,
    "joint_revision_success": 1.0,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def gold_oracle_prediction(scenario: dict[str, Any]) -> dict[str, Any]:
    gold = canonical_hidden_gold_fields(scenario["hidden_gold"])
    return {
        "scenario_id": scenario["scenario_id"],
        "response": {
            "decision": gold["expected_decision"],
            "memory_state": gold["expected_memory_state"],
            "evidence_event_ids": gold["expected_evidence_event_ids"],
            "failure_diagnosis": gold["expected_failure_diagnosis"],
            "answer": gold["expected_answer"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--out", default=None, help="Optional path to write metrics JSON report")
    args = parser.parse_args(argv)

    rows = read_jsonl(Path(args.data))
    scored_rows = []
    for scenario in rows:
        prediction = gold_oracle_prediction(scenario)
        gold = canonical_hidden_gold_fields(scenario["hidden_gold"])
        metrics = score_prediction(scenario, prediction)
        scored_rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "expected_decision": gold["expected_decision"],
                "response": prediction["response"],
                "metrics": metrics,
            }
        )

    aggregate = aggregate_metrics(scored_rows)
    all_metrics = aggregate.get("all_metrics") or aggregate.get("metrics") or {}

    failures: list[str] = []
    for key, expected in GOLD_ORACLE_EXPECTATIONS.items():
        actual = all_metrics.get(key)
        if actual is None:
            failures.append(f"missing metric {key}")
            continue
        if key == "stale_reuse_rate":
            if actual > expected + args.tolerance:
                failures.append(f"{key}={actual} expected <= {expected}")
        elif abs(actual - expected) > args.tolerance:
            failures.append(f"{key}={actual} expected {expected}")

    report = {
        "count": len(rows),
        "gold_oracle_metrics": {k: all_metrics.get(k) for k in GOLD_ORACLE_EXPECTATIONS},
        "all_metrics": all_metrics,
        "pass": not failures,
        "failures": failures,
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

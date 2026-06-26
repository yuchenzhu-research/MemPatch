"""Replay paired action corruptions through the deterministic Revision Guard."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._root import bootstrap_from

bootstrap_from(__file__)

from mempatch.benchmark.api import load_scenarios
from mempatch.revision.runtime.revision_module import run_revision_module_on_scenario


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _corruptions(actions_text: str) -> dict[str, str]:
    corruptions = {
        "malformed_json": actions_text[:-1] if actions_text else "{",
        "non_array_payload": "{}",
    }
    try:
        actions = json.loads(actions_text)
    except Exception:
        return corruptions
    if not isinstance(actions, list) or not actions:
        return corruptions

    unknown_evidence = copy.deepcopy(actions)
    unknown_evidence[0]["evidence_ids"] = ["EV_UNKNOWN"]
    corruptions["unknown_evidence"] = json.dumps(unknown_evidence)

    unknown_target = copy.deepcopy(actions)
    first = unknown_target[0]
    if first.get("target_belief_id") is not None:
        first["target_belief_id"] = "M_UNKNOWN"
    elif first.get("target_condition_id") is not None:
        first["target_condition_id"] = "C_UNKNOWN"
    else:
        first["action_type"] = "UNCERTAIN"
        first["target_belief_id"] = "M_UNKNOWN"
        first["target_condition_id"] = None
        first["replacement_belief_id"] = None
    corruptions["unknown_target"] = json.dumps(unknown_target)

    missing_field = copy.deepcopy(actions)
    missing_field[0].pop("evidence_ids", None)
    corruptions["missing_required_field"] = json.dumps(missing_field)
    return corruptions


def _rejected(prediction: dict[str, Any]) -> bool:
    audit = prediction.get("dpa_audit") or {}
    parse_result = audit.get("parse_result") or {}
    return (
        not parse_result.get("valid_json", False)
        or not parse_result.get("schema_valid", False)
        or bool(audit.get("rejected_actions"))
        or bool(audit.get("engine_errors"))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--raw-cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    scenarios = {str(row["scenario_id"]): row for row in load_scenarios(args.data)}
    rows = _rows(Path(args.raw_cases))
    if args.limit is not None:
        rows = rows[: args.limit]
    counts: Counter[str] = Counter()
    latencies: list[float] = []
    details = []
    for row in rows:
        sid = str(row["scenario_id"])
        scenario = scenarios[sid]
        generation = row["generations"]["mempatch_shared_actions"]
        raw_response = row["predictions"]["frozen_direct"]["response"]
        clean = run_revision_module_on_scenario(
            scenario,
            actions_text=generation["actions_text"],
            raw_response=raw_response,
            include_audit=True,
        )
        clean_accepted = not _rejected(clean)
        counts["clean_total"] += 1
        counts["clean_accepted"] += int(clean_accepted)
        for corruption, actions_text in _corruptions(generation["actions_text"]).items():
            started = time.perf_counter()
            first = run_revision_module_on_scenario(
                scenario,
                actions_text=actions_text,
                raw_response=raw_response,
                include_audit=True,
            )
            second = run_revision_module_on_scenario(
                scenario,
                actions_text=actions_text,
                raw_response=raw_response,
                include_audit=True,
            )
            latency = (time.perf_counter() - started) / 2
            latencies.append(latency)
            rejected = _rejected(first)
            deterministic = (
                first["response"] == second["response"]
                and first["dpa_audit"] == second["dpa_audit"]
            )
            counts["total"] += 1
            counts["rejected"] += int(rejected)
            counts["deterministic"] += int(deterministic)
            counts[f"{corruption}_total"] += 1
            counts[f"{corruption}_rejected"] += int(rejected)
            details.append(
                {
                    "scenario_id": sid,
                    "corruption": corruption,
                    "rejected": rejected,
                    "deterministic": deterministic,
                    "latency_seconds": latency,
                    "response": first["response"],
                    "audit": first["dpa_audit"],
                }
            )
    latencies.sort()

    def percentile(q: float) -> float:
        if not latencies:
            return 0.0
        return latencies[min(int(q * (len(latencies) - 1)), len(latencies) - 1)]

    report = {
        "cases": len(rows),
        "clean_acceptance_rate": counts["clean_accepted"] / max(counts["clean_total"], 1),
        "corrupted_proposals": counts["total"],
        "rejection_rate": counts["rejected"] / max(counts["total"], 1),
        "deterministic_replay_rate": counts["deterministic"] / max(counts["total"], 1),
        "latency_seconds_p50": percentile(0.50),
        "latency_seconds_p95": percentile(0.95),
        "by_corruption": {
            name: counts[f"{name}_rejected"] / max(counts[f"{name}_total"], 1)
            for name in (
                "malformed_json",
                "non_array_payload",
                "unknown_evidence",
                "unknown_target",
                "missing_required_field",
            )
        },
        "details": details,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Select a deterministic, stratified 30-case smoke subset from test500."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SEED = 20270614
DEFAULT_COUNT = 30


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stratum(row: dict[str, Any]) -> tuple[str, str] | None:
    gold = row.get("hidden_gold") or {}
    decision = gold.get("expected_decision") or row.get("decision")
    if decision:
        return "decision", str(decision)
    failure = (
        row.get("primary_failure_mode")
        or row.get("failure_mode")
        or gold.get("expected_failure_diagnosis")
    )
    if failure:
        return "failure_mode", str(failure)
    return None


def select_cases(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count > len(rows):
        raise ValueError(f"requested {count} smoke cases from only {len(rows)} scenarios")
    keyed = [(row, stratum(row)) for row in rows]
    available = [key for _, key in keyed if key is not None]
    if not available:
        return sorted(rows, key=lambda row: str(row["scenario_id"]))[:count]

    kind = "decision" if any(key[0] == "decision" for key in available) else "failure_mode"
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, key in keyed:
        if key and key[0] == kind:
            groups[key[1]].append(row)

    rng = random.Random(seed)
    for label in sorted(groups):
        groups[label].sort(key=lambda row: str(row["scenario_id"]))
        rng.shuffle(groups[label])

    selected: list[dict[str, Any]] = []
    labels = sorted(groups)
    while len(selected) < count:
        progressed = False
        for label in labels:
            if groups[label] and len(selected) < count:
                selected.append(groups[label].pop())
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        selected_ids = {str(row["scenario_id"]) for row in selected}
        remainder = sorted(
            (row for row in rows if str(row["scenario_id"]) not in selected_ids),
            key=lambda row: str(row["scenario_id"]),
        )
        selected.extend(remainder[: count - len(selected)])
    return sorted(selected, key=lambda row: str(row["scenario_id"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--ids-out", type=Path, default=Path("artifacts/smoke/smoke_case_ids.json"))
    parser.add_argument("--scenarios-out", type=Path, default=Path("artifacts/smoke/scenarios.jsonl"))
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    selected = select_cases(read_jsonl(args.input), args.count, args.seed)
    args.ids_out.parent.mkdir(parents=True, exist_ok=True)
    args.scenarios_out.parent.mkdir(parents=True, exist_ok=True)
    args.ids_out.write_text(
        json.dumps([row["scenario_id"] for row in selected], indent=2) + "\n",
        encoding="utf-8",
    )
    with args.scenarios_out.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Selected {len(selected)} deterministic smoke cases -> {args.ids_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

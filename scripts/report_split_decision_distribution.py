#!/usr/bin/env python3
"""Report hidden_gold.expected_decision distribution for MemPatch scenario splits."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.mempatch_bench.general_taxonomy import DECISIONS
from benchmark.mempatch_bench.general_taxonomy import canonical_hidden_gold_fields


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_no}: invalid JSON: {exc}") from exc
    return rows


def resolve_scenarios_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "scenarios.jsonl"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"no scenarios.jsonl in directory: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"scenarios file not found: {path}")
    return path


def decision_distribution(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        gold = canonical_hidden_gold_fields(row.get("hidden_gold") or {})
        decision = gold.get("expected_decision") or "<missing>"
        counts[decision] += 1
    return counts


def renderer_distribution(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("metadata", {}).get("renderer") or "<missing>") for row in rows)


def pattern_distribution(rows: list[dict[str, Any]], decision: str, *, top: int = 8) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for row in rows:
        gold = canonical_hidden_gold_fields(row.get("hidden_gold") or {})
        if gold.get("expected_decision") != decision:
            continue
        counts[str(row.get("pattern") or "<missing>")] += 1
    return counts.most_common(top)


def load_manifest_quotas(manifest_path: Path, split_name: str) -> dict[str, int] | None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    quotas = payload.get("decision_quotas", {})
    split_quotas = quotas.get(split_name)
    if not isinstance(split_quotas, dict):
        return None
    return {str(k): int(v) for k, v in split_quotas.items()}


def check_quotas(
    observed: Counter[str],
    expected: dict[str, int],
    *,
    split_name: str,
) -> list[str]:
    errors: list[str] = []
    for decision in DECISIONS:
        need = expected.get(decision, 0)
        have = observed.get(decision, 0)
        if need > 0 and have < need:
            errors.append(f"{split_name}: {decision} need>={need}, have={have}")
        if need == 0 and have > 0 and decision in ("ask_clarification", "escalate"):
            pass
    for decision, count in observed.items():
        if decision not in DECISIONS and decision != "<missing>":
            errors.append(f"{split_name}: unknown decision label {decision!r} (n={count})")
    missing_labels = [d for d in DECISIONS if expected.get(d, 0) > 0 and observed.get(d, 0) == 0]
    if missing_labels:
        errors.append(f"{split_name}: missing required labels: {missing_labels}")
    return errors


def report_split(
    split_name: str,
    path: Path,
    *,
    manifest_path: Path | None,
    top_patterns: int,
) -> dict[str, Any]:
    rows = read_jsonl(resolve_scenarios_path(path))
    decisions = decision_distribution(rows)
    renderers = renderer_distribution(rows)
    payload: dict[str, Any] = {
        "split": split_name,
        "path": str(path),
        "count": len(rows),
        "decision_counts": {d: decisions.get(d, 0) for d in DECISIONS},
        "extra_decisions": {
            k: v for k, v in sorted(decisions.items()) if k not in DECISIONS
        },
        "renderers": dict(renderers),
        "scenario_id_range": (
            rows[0].get("scenario_id"),
            rows[-1].get("scenario_id"),
        )
        if rows
        else None,
    }
    print(f"\n== {split_name} ({len(rows)} rows) ==")
    print(f"path: {path}")
    if rows:
        print(f"scenario_id: {payload['scenario_id_range'][0]} .. {payload['scenario_id_range'][1]}")
    print("expected_decision:")
    for decision in DECISIONS:
        if decisions.get(decision, 0):
            print(f"  {decision}: {decisions[decision]}")
    for decision, count in sorted(decisions.items()):
        if decision not in DECISIONS:
            print(f"  {decision}: {count}")
    if renderers:
        print(f"renderers: {dict(renderers)}")
    for decision in ("ask_clarification", "escalate"):
        if decisions.get(decision, 0):
            print(f"  top patterns for {decision}: {pattern_distribution(rows, decision, top=top_patterns)}")

    if manifest_path is not None:
        expected = load_manifest_quotas(manifest_path, split_name)
        if expected:
            errors = check_quotas(decisions, expected, split_name=split_name)
            payload["quota_errors"] = errors
            if errors:
                print("quota check: FAIL")
                for err in errors:
                    print(f"  - {err}")
            else:
                print("quota check: OK")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report MemPatch split decision distributions.")
    parser.add_argument(
        "--split",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        required=True,
        help="Split name and scenarios.jsonl path or directory (repeatable)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional v1.2 manifest with decision_quotas to validate against",
    )
    parser.add_argument("--top-patterns", type=int, default=8)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write machine-readable summary JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summaries: dict[str, Any] = {}
    exit_code = 0
    for split_name, raw_path in args.split:
        path = Path(raw_path)
        summary = report_split(
            split_name,
            path,
            manifest_path=args.manifest,
            top_patterns=args.top_patterns,
        )
        summaries[split_name] = summary
        if summary.get("quota_errors"):
            exit_code = 1

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summaries, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"\nWrote summary to {args.json_out}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

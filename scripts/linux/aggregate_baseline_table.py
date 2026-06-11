#!/usr/bin/env python3
"""Aggregate saved baseline metrics JSON into a markdown table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._root import bootstrap_from

bootstrap_from(__file__)

from scripts.memory.context_builders import (  # noqa: E402
    BASELINE_DISPLAY_NAMES,
    DIAGNOSTIC_UPPER_BOUND_IDS,
)

PAPER_METRICS = (
    "decision_macro_f1",
    "memory_state_accuracy",
    "response_schema_compliance_rate",
    "evidence_f1",
    "failure_diagnosis_accuracy",
    "joint_revision_success",
    "stale_reuse_rate",
)


def load_metrics(path: Path) -> tuple[dict, dict] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("headline_metrics") or payload
    return metrics, payload.get("run_meta") or {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    rows: list[tuple[str, dict]] = []
    for metrics_path in sorted(args.results_dir.glob("*_metrics.json")):
        name = metrics_path.name.replace("_metrics.json", "")
        if name.removeprefix("baseline_") in DIAGNOSTIC_UPPER_BOUND_IDS:
            continue
        loaded = load_metrics(metrics_path)
        if not loaded:
            continue
        metrics, run_meta = loaded
        baseline_id = run_meta.get("baseline")
        display_name = BASELINE_DISPLAY_NAMES.get(baseline_id, name)
        rows.append((display_name, metrics))

    if not rows:
        print(f"no metrics under {args.results_dir}", file=sys.stderr)
        return 1

    lines = [
        "| method | " + " | ".join(PAPER_METRICS) + " |",
        "|" + "|".join(["---"] * (len(PAPER_METRICS) + 1)) + "|",
    ]
    for name, metrics in rows:
        cells = [f"{metrics.get(k, 0.0):.3f}" for k in PAPER_METRICS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

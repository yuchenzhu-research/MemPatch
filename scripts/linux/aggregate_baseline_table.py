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

from benchmark.api import HEADLINE_METRICS  # noqa: E402


def load_metrics(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("headline_metrics") or payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    rows: list[tuple[str, dict]] = []
    for metrics_path in sorted(args.results_dir.glob("*_metrics.json")):
        metrics = load_metrics(metrics_path)
        if not metrics:
            continue
        name = metrics_path.name.replace("_metrics.json", "")
        rows.append((name, metrics))

    if not rows:
        print(f"no metrics under {args.results_dir}", file=sys.stderr)
        return 1

    lines = [
        "| method | " + " | ".join(HEADLINE_METRICS) + " |",
        "|" + "|".join(["---"] * (len(HEADLINE_METRICS) + 1)) + "|",
    ]
    for name, metrics in rows:
        cells = [f"{metrics.get(k, 0.0):.3f}" for k in HEADLINE_METRICS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

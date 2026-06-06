#!/usr/bin/env python3
"""Run the MemPatch Revision Module full method path and write predictions JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.retrace_bench.api import load_scenarios  # noqa: E402
from retrace_learn.runtime.revision_module import run_revision_module_on_scenario  # noqa: E402


def _load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                sid = row.get("scenario_id")
                if sid:
                    done.add(str(sid))
    return done


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MemPatch Revision Module: scenario → view → policy → "
            "DPA projection → benchmark response."
        ),
    )
    parser.add_argument(
        "--data",
        required=True,
        help="scenarios.jsonl file or directory containing scenarios.jsonl",
    )
    parser.add_argument(
        "--out-predictions",
        required=True,
        help="Path to write canonical predictions JSONL",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional maximum number of new scenarios to run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip scenario_ids already present in --out-predictions",
    )
    parser.add_argument(
        "--policy",
        choices=("noop",),
        default="noop",
        help="Revision Response Policy variant (default: noop smoke policy)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scenarios = load_scenarios(args.data)
    out_path = Path(args.out_predictions)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = _load_done_ids(out_path) if args.resume else set()
    mode = "a" if args.resume and out_path.exists() else "w"
    planned = [s for s in scenarios if s["scenario_id"] not in done]
    if args.max_cases is not None:
        planned = planned[: args.max_cases]

    print(
        f"MemPatch Revision Module runner | policy={args.policy} | "
        f"planned={len(planned)} | resume={args.resume}"
    )

    with out_path.open(mode, encoding="utf-8") as out_f:
        for scenario in planned:
            prediction = run_revision_module_on_scenario(scenario)
            out_f.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    print(f"wrote {len(planned)} predictions to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

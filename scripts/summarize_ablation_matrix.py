#!/usr/bin/env python3
"""Summarize Path A / Path B ablation metrics into comparison tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HEADLINE_KEYS = (
    "joint_revision_success",
    "decision_macro_f1",
    "memory_state_accuracy",
    "evidence_f1",
    "failure_diagnosis_accuracy",
)


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing *_metrics.json files from ablation runs.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to write the aggregated matrix JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = sorted(args.results_dir.rglob("*_metrics.json"))
    if not files:
        print(f"no metrics files under {args.results_dir}", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    for path in files:
        payload = load_metrics(path)
        headline = payload.get("headline_metrics") or {}
        rows.append(
            {
                "file": path.name,
                "model": payload.get("model") or path.name.split("_")[0],
                "path": payload.get("path") or "?",
                "variant": payload.get("variant") or "?",
                "split": payload.get("split") or "?",
                "count": payload.get("count"),
                **{key: headline.get(key) for key in HEADLINE_KEYS},
            }
        )

    print("\n== Ablation matrix (headline metrics) ==")
    header = f"{'model':<14} {'path':<5} {'variant':<8} {'split':<4} " + " ".join(
        f"{k[:6]:>7}" for k in HEADLINE_KEYS
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['model']:<14} {row['path']:<5} {row['variant']:<8} {row['split']:<4} "
            + " ".join(f"{float(row.get(k, 0.0) or 0.0):7.3f}" for k in HEADLINE_KEYS)
        )

    print("\n== Same-model ablation (Path B LoRA vs Path A base) ==")
    by_key: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        by_key[(row["model"], row["split"])] = by_key.get((row["model"], row["split"]), {})
        tag = f"{row['path']}_{row['variant']}"
        by_key[(row["model"], row["split"])][tag] = float(row.get("joint_revision_success") or 0.0)
    for (model, split), metrics in sorted(by_key.items()):
        b = metrics.get("B_lora", metrics.get("B_base"))
        a = metrics.get("A_base")
        if b is None or a is None:
            continue
        print(f"{model} {split}: path_b={b:.3f} path_a={a:.3f} delta={a - b:+.3f}")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote matrix -> {args.out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

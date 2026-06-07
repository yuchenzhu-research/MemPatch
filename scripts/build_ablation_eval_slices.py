#!/usr/bin/env python3
"""Build balanced L3/L4 scenario slices for ablation eval (Path A and Path B)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark.general_taxonomy import DECISIONS  # noqa: E402
from prepare_mempatch_v13_smoke import (  # noqa: E402
    HARD_QUOTAS,
    index_by_decision,
    read_jsonl,
    sample_quotas,
    scenario_decision,
    sft_example,
    write_jsonl,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=root / "hf_release/mempatch/validation/scenarios.jsonl",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=root / "hf_release/mempatch/test/scenarios.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "local/train_data/ablation",
    )
    parser.add_argument("--l3-count", type=int, default=50, help="Total L3 validation rows.")
    parser.add_argument("--l4-count", type=int, default=50, help="Total L4 test rows.")
    parser.add_argument("--seed", type=int, default=20270607)
    return parser.parse_args(argv)


def per_decision_quota(total: int) -> dict[str, int]:
    if total % len(DECISIONS) != 0:
        raise ValueError(f"{total} cases must be divisible by {len(DECISIONS)} decision labels")
    each = total // len(DECISIONS)
    return {decision: each for decision in DECISIONS}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    l3_quota = per_decision_quota(args.l3_count)
    l4_quota = per_decision_quota(args.l4_count) if args.l4_count == len(DECISIONS) * 10 else None
    if l4_quota is None:
        # Keep the same 10-per-label hard slice shape when l4-count is 50.
        l4_quota = dict(HARD_QUOTAS)

    valid_rows = read_jsonl(args.validation_data)
    test_rows = read_jsonl(args.test_data)

    l3_sampled, l3_actual = sample_quotas(
        index_by_decision(valid_rows),
        l3_quota,
        seed=args.seed,
        split_name="validation",
    )
    l4_sampled, l4_actual = sample_quotas(
        index_by_decision(test_rows),
        l4_quota,
        seed=args.seed + 1,
        split_name="test",
    )

    l3_dir = args.out_dir / "eval_l3_valid50"
    l4_dir = args.out_dir / "eval_l4_hard50"
    write_jsonl(l3_dir / "scenarios.jsonl", l3_sampled)
    write_jsonl(l4_dir / "scenarios.jsonl", l4_sampled)
    write_jsonl(l3_dir / "sft.jsonl", [sft_example(row) for row in l3_sampled])
    write_jsonl(l4_dir / "sft.jsonl", [sft_example(row) for row in l4_sampled])

    manifest = {
        "seed": args.seed,
        "l3": {
            "count": len(l3_sampled),
            "requested_quota": l3_quota,
            "actual_quota": l3_actual,
            "split": "validation",
            "difficulty": "L3",
            "scenario_ids": [row["scenario_id"] for row in l3_sampled],
        },
        "l4": {
            "count": len(l4_sampled),
            "requested_quota": l4_quota,
            "actual_quota": l4_actual,
            "split": "test",
            "difficulty": "L4",
            "scenario_ids": [row["scenario_id"] for row in l4_sampled],
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "eval_slices.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote L3 slice ({len(l3_sampled)}) -> {l3_dir / 'scenarios.jsonl'}")
    print(f"Wrote L4 slice ({len(l4_sampled)}) -> {l4_dir / 'scenarios.jsonl'}")
    print(f"Wrote manifest -> {args.out_dir / 'eval_slices.json'}")
    for label, rows in ("L3", l3_sampled), ("L4", l4_sampled):
        counts = {d: 0 for d in DECISIONS}
        for row in rows:
            decision = scenario_decision(row)
            if decision in counts:
                counts[decision] += 1
        print(f"{label} decision counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

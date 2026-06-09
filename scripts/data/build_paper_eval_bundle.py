#!/usr/bin/env python3
"""Build Path-B SFT JSONL + scenario JSONL bundles for paper eval splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_mempatch_v13_smoke import read_jsonl, sft_example, write_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=root / "hf_release/mempatch/test/scenarios.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "local/train_data/paper/test500",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_jsonl(args.scenarios)
    if args.limit is not None:
        rows = rows[: args.limit]
    write_jsonl(args.out_dir / "scenarios.jsonl", rows)
    write_jsonl(args.out_dir / "sft.jsonl", [sft_example(row) for row in rows])
    manifest = {
        "source": str(args.scenarios),
        "count": len(rows),
        "split": rows[0].get("public_split_name") if rows else None,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} scenarios + sft rows -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

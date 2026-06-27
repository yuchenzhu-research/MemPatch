#!/usr/bin/env python3
"""Package MemPatch-Bench v1.4 raw splits into public/label release files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools._root import bootstrap_from

bootstrap_from(__file__)

from mempatch.benchmark.release import export_release


def discover_splits(input_dir: Path, splits: tuple[str, ...]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for split in splits:
        candidates = (
            input_dir / f"{split}.jsonl",
            input_dir / split / "scenarios.jsonl",
        )
        for candidate in candidates:
            if candidate.is_file():
                found[split] = candidate
                break
    return found


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package MemPatch-Bench v1.4 release files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing SPLIT.jsonl or SPLIT/scenarios.jsonl raw internal files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Release bundle directory containing public/, labels/, and manifests/.",
    )
    parser.add_argument("--splits", default="dev_calibration,main_test_synthetic,challenge_test_hard")
    parser.add_argument("--release-version", default="v1.4.0-dev")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    split_names = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    split_paths = discover_splits(args.input_dir, split_names)
    if not split_paths:
        print(
            f"error: no raw split JSONL found under {args.input_dir} for {{{','.join(split_names)}}}",
            file=sys.stderr,
        )
        return 1

    manifest = export_release(split_paths, args.out_dir, version=args.release_version)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

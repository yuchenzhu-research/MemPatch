#!/usr/bin/env python3
"""Generate MemPatch-Bench v1.4 synthetic raw data and optional release views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._root import bootstrap_from

bootstrap_from(__file__)

from mempatch.benchmark.generate import DEFAULT_QUOTAS, generate_raw_files, validate_generated_row
from mempatch.benchmark.release import export_release, read_jsonl


def _quota(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SPLIT=COUNT")
    split, raw_count = value.split("=", 1)
    try:
        count = int(raw_count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("COUNT must be an integer") from exc
    if count < 0:
        raise argparse.ArgumentTypeError("COUNT must be non-negative")
    return split, count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MemPatch-Bench v1.4 synthetic scenarios.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("local/data/mempatch/v1.4/raw_internal"),
        help="Output directory for raw internal split JSONL.",
    )
    parser.add_argument(
        "--quota",
        action="append",
        type=_quota,
        default=[],
        help="Override split count as SPLIT=COUNT. Repeatable.",
    )
    parser.add_argument(
        "--seed-namespace",
        default="mempatch_v14",
        help="Deterministic seed namespace.",
    )
    parser.add_argument(
        "--release-out",
        type=Path,
        default=None,
        help="Optional output directory for public/labels/manifests release bundle.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Optional path for generation manifest JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    quotas = dict(args.quota) if args.quota else DEFAULT_QUOTAS
    paths = generate_raw_files(args.out_dir, quotas, seed_namespace=args.seed_namespace)
    validation_errors: list[str] = []
    for path in paths.values():
        for row in read_jsonl(path):
            validation_errors.extend(validate_generated_row(row))
    if validation_errors:
        print(f"Validation FAILED ({len(validation_errors)} errors):", file=sys.stderr)
        for err in validation_errors[:20]:
            print(f"  - {err}", file=sys.stderr)
        if len(validation_errors) > 20:
            print(f"  ... +{len(validation_errors) - 20} more", file=sys.stderr)
        return 4

    manifest = {
        "generator": "mempatch.benchmark.generate",
        "schema_version": "mempatch_bench_v1.4",
        "seed_namespace": args.seed_namespace,
        "raw_output_dir": str(args.out_dir),
        "splits": {split: {"path": str(path), "count": quotas[split]} for split, path in paths.items()},
        "release_output_dir": None,
    }

    if args.release_out:
        release_manifest = export_release(paths, args.release_out)
        manifest["release_output_dir"] = str(args.release_out)
        manifest["release_manifest"] = release_manifest

    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    if args.manifest_out:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote manifest to {args.manifest_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

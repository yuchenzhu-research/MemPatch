#!/usr/bin/env python3
"""Generate MemPatch v1.3 pilot/full JSONL (train/validation/test)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.generation.blueprints import RENDERER, validate_registry
from benchmark.generation.export_jsonl import export_splits
from benchmark.generation.scenario_builder import build_scenario
from benchmark.generation.split_sampler import FULL_QUOTAS, PILOT_QUOTAS, pilot_blueprints, sample_split
from scripts.validate_mempatch_bench_dataset import validate_one


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MemPatch v1.3 pilot scenarios.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("hf_release/mempatch"),
        help="Output directory for split JSONL (default: hf_release/mempatch)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full quotas (train 2700 / validation 800 / test 500) instead of pilot",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate registry and print sampling plan without rendering",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Optional path for export manifest JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    registry_errors = validate_registry()
    if registry_errors:
        print("Registry validation FAILED:", file=sys.stderr)
        for err in registry_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print("Registry validation: OK")

    quotas = FULL_QUOTAS if args.full else PILOT_QUOTAS
    samples = pilot_blueprints() if not args.full else [
        item for split, split_quotas in quotas.items() for item in sample_split(split, split_quotas)
    ]

    plan = {
        "renderer": RENDERER,
        "mode": "full" if args.full else "pilot",
        "total_blueprints": len(samples),
        "by_split": {},
    }
    for split in quotas:
        split_samples = [s for s in samples if s.blueprint.split == split]
        plan["by_split"][split] = {
            "count": len(split_samples),
            "quotas": quotas[split],
            "variants": sorted({s.variant.variant_id for s in split_samples}),
        }
    print(json.dumps(plan, indent=2))

    if args.dry_run:
        print("Dry run complete — no scenarios rendered.")
        return 0

    scenarios_by_split: dict[str, list] = {split: [] for split in quotas}
    validation_errors: list[str] = []
    for sampled in samples:
        result = build_scenario(
            blueprint=sampled.blueprint,
            variant=sampled.variant,
            family=sampled.family,
        )
        errors, _warnings = validate_one(result.scenario, packaging_final=False)
        validation_errors.extend(errors)
        scenarios_by_split[sampled.blueprint.split].append(result.scenario)

    if validation_errors:
        print(f"Validation FAILED ({len(validation_errors)} errors):", file=sys.stderr)
        for err in validation_errors[:20]:
            print(f"  - {err}", file=sys.stderr)
        if len(validation_errors) > 20:
            print(f"  ... +{len(validation_errors) - 20} more", file=sys.stderr)
        return 4

    manifest = export_splits(scenarios_by_split, args.out_dir)
    print(f"Exported {manifest['splits']}")

    if args.manifest_out:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote manifest to {args.manifest_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

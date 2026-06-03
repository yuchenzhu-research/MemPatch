#!/usr/bin/env python3
"""Generate the final AAAI paper-facing ReTrace-Bench splits.

Generates:
  - main (3000)
  - hard (500)
  - realistic (200)
  - calibration (80)
  - private_hidden (200)

Usage::
    PYTHONPATH=. python scripts/generate_retrace_bench_final.py \
      --seed 2027 \
      --github-seeds data/source_seeds/github_workflow_seeds.jsonl \
      --out data/retrace_bench
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.retrace_bench.generation.github_workflow_seeds import load_seeds_from_jsonl
from benchmark.retrace_bench.generation.hard_plus_blueprints import build_deterministic_scenario
from benchmark.retrace_bench.generation.github_realistic_blueprints import build_github_realistic_scenario
from benchmark.retrace_bench.generation.release_manifest import build_manifest


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate final paper-facing dataset splits.")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--github-seeds", default="data/source_seeds/github_workflow_seeds.jsonl")
    parser.add_argument("--out", default="data/retrace_bench")
    parser.add_argument("--smoke", action="store_true", help="Generate small smoke datasets for verification")
    args = parser.parse_args()

    seed_path = Path(args.github_seeds)
    out_base = Path(args.out)
    
    # Load mined GitHub workflow seeds if available
    github_seeds = load_seeds_from_jsonl(seed_path)
    print(f"Loaded {len(github_seeds)} GitHub seeds.")

    # Configure splits
    splits_config = {
        "main": 3000 if not args.smoke else 30,
        "hard": 500 if not args.smoke else 30,
        "realistic": 200 if not args.smoke else 20,
        "calibration": 80 if not args.smoke else 20,
        "private_hidden": 200 if not args.smoke else 20
    }

    # For each split, generate, validate, and write scenarios
    for split_name, count in splits_config.items():
        print(f"Generating split '{split_name}' with {count} scenarios...")
        rng = random.Random(args.seed + hash(split_name))
        
        scenarios = []
        for i in range(count):
            if split_name == "realistic" and github_seeds:
                # Use github seed for realistic split
                seed_obj = github_seeds[i % len(github_seeds)]
                sc = build_github_realistic_scenario(i, seed_obj, split_name, args.seed)
            else:
                # Use deterministic builder for others
                sc = build_deterministic_scenario(i, split_name, args.seed)
            
            scenarios.append(sc)
            
        # Shuffling to prevent any first-N bias and sorting by ID
        scenarios.sort(key=lambda s: s["scenario_id"])
        
        # Build split directory
        split_dir = out_base / f"{split_name}_{count}_en"
        if split_name == "private_hidden":
            # Keep hidden golds private in actual packaging (will be stripped during public rendering)
            split_dir = out_base / f"private_hidden_{count}_en"
            
        write_jsonl(split_dir / "scenarios.jsonl", scenarios)
        
        # Build manifest
        source_type = "controlled_synthetic"
        if split_name == "realistic":
            source_type = "github_realistic"
            
        manifest = build_manifest(
            scenarios,
            split=split_name,
            source_type=source_type,
            annotation_status="reviewed" if split_name == "realistic" else "synthetic_gold",
            role=f"Paper-facing final {split_name} benchmark split."
        )
        
        # Inject additional metadata to fulfill AAAI requirements
        manifest["benchmark_release_name"] = "final_aaai"
        manifest["generation_commit"] = "final_release"
        manifest["random_seed"] = args.seed
        manifest["intended_use"] = "AAAI ReTrace-Bench paper evaluation baseline reference"
        manifest["not_for_training_notice"] = "Strictly evaluation-only. Never train ReTrace-Learn on this data."
        manifest["calibration_is_smoke_only_notice"] = "Calibration split is for sanity checks only."
        
        write_manifest(split_dir / "manifest.json", manifest)
        
        # Create an empty or brief README.md for the split folder
        readme_text = f"# ReTrace-Bench `{split_name}_{count}_en` (Final release)\n\n"
        readme_text += f"Public Split Name: **`{split_name}`**\n"
        readme_text += f"Count: {count}\n"
        readme_text += f"Role: {manifest['role']}\n\n"
        readme_text += "## CONTAMINATION WARNING\n"
        readme_text += "Do NOT use this evaluation dataset for training models.\n"
        split_dir.joinpath("README.md").write_text(readme_text, encoding="utf-8")
        
        print(f"Split '{split_name}' successfully generated.")

    print("All splits successfully created.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

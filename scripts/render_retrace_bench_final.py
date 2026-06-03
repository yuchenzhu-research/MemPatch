#!/usr/bin/env python3
"""Render and filter final ReTrace-Bench dataset splits for release packaging.

For public packaging:
  - 'private_hidden' split will have 'hidden_gold' stripped to prevent leakage.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Render and package splits for release.")
    parser.add_argument("--src", default="data/retrace_bench")
    parser.add_argument("--dest", default="data/retrace_bench_release")
    args = parser.parse_args()

    src_base = Path(args.src)
    dest_base = Path(args.dest)
    dest_base.mkdir(parents=True, exist_ok=True)

    # Walk through split folders in source
    for split_dir in src_base.glob("*_en"):
        if not split_dir.is_dir():
            continue
        
        split_name = split_dir.name.split("_")[0]
        print(f"Rendering split directory: {split_dir.name}...")
        
        scenarios_file = split_dir / "scenarios.jsonl"
        manifest_file = split_dir / "manifest.json"
        
        if not scenarios_file.exists():
            continue
            
        dest_split_dir = dest_base / split_dir.name
        dest_split_dir.mkdir(parents=True, exist_ok=True)
        
        # Read and optionally strip hidden gold for private_hidden
        output_scenarios = []
        with scenarios_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                sc = json.loads(line)
                
                # Check for private split
                if split_name == "private_hidden":
                    # Remove gold keys to prevent leakage
                    if "hidden_gold" in sc:
                        del sc["hidden_gold"]
                
                output_scenarios.append(sc)
                
        # Write to destination
        dest_scenarios = dest_split_dir / "scenarios.jsonl"
        with dest_scenarios.open("w", encoding="utf-8") as f:
            for sc in output_scenarios:
                f.write(json.dumps(sc, ensure_ascii=False) + "\n")
                
        # Copy manifest
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            dest_manifest = dest_split_dir / "manifest.json"
            dest_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            
        # Copy readme
        src_readme = split_dir / "README.md"
        if src_readme.exists():
            (dest_split_dir / "README.md").write_text(src_readme.read_text(encoding="utf-8"), encoding="utf-8")

        print(f"Split {split_name} successfully rendered to {dest_split_dir.name}")

    print("All release splits rendered successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

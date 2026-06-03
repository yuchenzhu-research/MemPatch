import os
import json
import argparse
import sys
from benchmark.retrace_bench.schemas_v2 import manifest_from_dict, scenario_from_dict
from benchmark.retrace_bench.validation_v2 import validate_scenario_v2, validate_manifest_v2


def main():
    parser = argparse.ArgumentParser(description="Validate ReTrace-Bench v2 Dataset")
    parser.add_argument("--data", required=True, help="Path to the dataset directory (e.g., data/retrace_bench/sample_20_v2)")
    args = parser.parse_args()

    data_dir = args.data
    manifest_path = os.path.join(data_dir, "manifest.json")
    scenarios_path = os.path.join(data_dir, "scenarios.jsonl")

    if not os.path.exists(manifest_path):
        print(f"Error: manifest.json not found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(scenarios_path):
        print(f"Error: scenarios.jsonl not found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    # 1. Load and validate manifest
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        manifest = manifest_from_dict(manifest_data)
        validate_manifest_v2(manifest)
        print("✓ manifest.json is valid.")
    except Exception as e:
        print(f"Manifest validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Load and validate scenarios
    scenarios = []
    errors = []
    scenarios_count = 0
    tasks_count = 0

    try:
        with open(scenarios_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    s_data = json.loads(line)
                    scenario = scenario_from_dict(s_data)
                    validate_scenario_v2(scenario)
                    scenarios.append(scenario)
                    scenarios_count += 1
                    tasks_count += len(scenario.tasks)
                except Exception as e:
                    errors.append(f"Line {line_idx} in scenarios.jsonl validation failed: {e}")
    except Exception as e:
        print(f"Failed to read scenarios.jsonl: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Print report
    print("\n--- Validation Report ---")
    print(f"Dataset Name: {manifest.dataset_name}")
    print(f"Version:      {manifest.version}")
    print(f"Scenarios:    {scenarios_count}")
    print(f"Tasks:        {tasks_count}")

    if errors:
        print(f"\n❌ Validation FAILED with {len(errors)} error(s):", file=sys.stderr)
        for err in errors:
            print(f"- {err}", file=sys.stderr)
        sys.exit(1)
    else:
        print("\n✓ All scenarios are valid and conform to schema v2!")
        sys.exit(0)


if __name__ == "__main__":
    main()

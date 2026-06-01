import os
import json
from benchmark.retrace_bench.schemas_v2 import scenario_from_dict, manifest_from_dict
from benchmark.retrace_bench.validation_v2 import validate_scenario_v2, validate_manifest_v2


def test_sample_20_v2_validation():
    # Construct paths relative to ReTrace root directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data", "retrace_bench", "sample_20_v2")

    manifest_path = os.path.join(data_dir, "manifest.json")
    scenarios_path = os.path.join(data_dir, "scenarios.jsonl")

    assert os.path.exists(manifest_path), f"Manifest file not found: {manifest_path}"
    assert os.path.exists(scenarios_path), f"Scenarios file not found: {scenarios_path}"

    # Validate manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    manifest = manifest_from_dict(manifest_data)
    validate_manifest_v2(manifest)

    # Validate scenarios
    scenarios_count = 0
    with open(scenarios_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s_dict = json.loads(line)
            scenario = scenario_from_dict(s_dict)
            validate_scenario_v2(scenario)
            scenarios_count += 1

    assert scenarios_count == 4
    assert len(manifest.scenarios) == 4

#!/usr/bin/env python3
"""
Package the ReTrace-Bench dataset for Hugging Face.
Copies canonical files from data/ and docs/ to release/huggingface/ReTrace-Bench/.
"""

import argparse
import os
import shutil
import json


SCENARIO_JSONL_NAME = "scenarios.jsonl"


def to_viewer_row(scenario):
    """Flatten nested scenario payloads so the Hugging Face viewer can infer a stable schema."""
    row = {
        "scenario_id": scenario["scenario_id"],
        "domain": scenario["domain"],
        "primary_failure_mode": scenario["primary_failure_mode"],
        "secondary_failure_modes_json": json.dumps(
            scenario.get("secondary_failure_modes", []),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "difficulty": scenario["difficulty"],
        "workflow_context": scenario["workflow_context"],
        "public_input_json": json.dumps(
            scenario.get("public_input", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "tasks_json": json.dumps(
            scenario.get("tasks", []),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "hidden_gold_json": json.dumps(
            scenario.get("hidden_gold", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "metadata_json": json.dumps(
            scenario.get("metadata", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    if "training_targets" in scenario:
        row["training_targets_json"] = json.dumps(
            scenario["training_targets"],
            ensure_ascii=False,
            sort_keys=True,
        )
    else:
        row["training_targets_json"] = ""
    return row


def copy_jsonl_for_viewer(src_path, tgt_path):
    with open(src_path, "r", encoding="utf-8") as src, open(
        tgt_path, "w", encoding="utf-8"
    ) as tgt:
        for line in src:
            if not line.strip():
                continue
            scenario = json.loads(line)
            tgt.write(json.dumps(to_viewer_row(scenario), ensure_ascii=False) + "\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Package ReTrace-Bench dataset for Hugging Face.")
    parser.add_argument(
        "--include-supervision",
        action="store_true",
        help="Include synthetic supervision/dev selection pools in the HF dataset package."
    )
    return parser.parse_args()

def count_scenarios(jsonl_path):
    if not os.path.exists(jsonl_path):
        return 0
    count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count

def main():
    args = parse_args()
    
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    hf_dir = os.path.join(repo_root, "release", "huggingface", "ReTrace-Bench")
    
    # 1. Clean output directory
    if os.path.exists(hf_dir):
        print(f"Cleaning existing Hugging Face package directory at {hf_dir}...")
        shutil.rmtree(hf_dir)
    os.makedirs(hf_dir, exist_ok=True)
    
    # Define mapping: (source_rel_path, target_rel_path)
    copy_manifest = []
    
    # Benchmark split
    copy_manifest.extend([
        ("data/retrace_bench/test_800_templateheldout_en/scenarios.jsonl", "benchmark/test_800_templateheldout_en/scenarios.jsonl"),
        ("data/retrace_bench/test_800_templateheldout_en/manifest.json", "benchmark/test_800_templateheldout_en/manifest.json"),
        ("data/retrace_bench/test_800_templateheldout_en/README.md", "benchmark/test_800_templateheldout_en/README.md"),
    ])
    
    # Calibration split
    copy_manifest.extend([
        ("data/retrace_bench/sample_80_hard_en/scenarios.jsonl", "calibration/sample_80_hard_en/scenarios.jsonl"),
        ("data/retrace_bench/sample_80_hard_en/manifest.json", "calibration/sample_80_hard_en/manifest.json"),
        ("data/retrace_bench/sample_80_hard_en/README.md", "calibration/sample_80_hard_en/README.md"),
    ])
    
    # Supervision splits (if requested)
    if args.include_supervision:
        copy_manifest.extend([
            ("data/retrace_supervision/train_3000_en/scenarios.jsonl", "supervision/train_3000_en/scenarios.jsonl"),
            ("data/retrace_supervision/train_3000_en/manifest.json", "supervision/train_3000_en/manifest.json"),
            ("data/retrace_supervision/train_3000_en/README.md", "supervision/train_3000_en/README.md"),
            ("data/retrace_supervision/dev_400_en/scenarios.jsonl", "supervision/dev_400_en/scenarios.jsonl"),
            ("data/retrace_supervision/dev_400_en/manifest.json", "supervision/dev_400_en/manifest.json"),
            ("data/retrace_supervision/dev_400_en/README.md", "supervision/dev_400_en/README.md"),
        ])
    
    # Docs mapping
    copy_manifest.extend([
        ("docs/retrace_bench/baseline_results_test_800_templateheldout_en.md", "docs/baseline_results_test_800_templateheldout_en.md"),
        ("docs/retrace_bench/template_signature_report.md", "docs/template_signature_report.md"),
        ("docs/retrace_bench/template_lookup_test_800_templateheldout_en.md", "docs/template_lookup_test_800_templateheldout_en.md"),
        ("docs/retrace_bench/split_leakage_report.md", "docs/split_leakage_report.md"),
        ("docs/project_governance.md", "docs/project_governance.md"),
    ])
    
    print("Copying files to Hugging Face release structure...")
    for src_rel, tgt_rel in copy_manifest:
        src_path = os.path.join(repo_root, src_rel)
        tgt_path = os.path.join(hf_dir, tgt_rel)
        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
            if os.path.basename(src_path) == SCENARIO_JSONL_NAME:
                copy_jsonl_for_viewer(src_path, tgt_path)
                print(f"  Converted for HF viewer: {src_rel} -> {tgt_rel}")
            else:
                shutil.copy2(src_path, tgt_path)
                print(f"  Copied: {src_rel} -> {tgt_rel}")
        else:
            # calibration manifest or readme might be optional based on task request
            if "manifest.json" in src_rel or "README.md" in src_rel:
                print(f"  Warning: Optional file not found: {src_rel}")
            else:
                raise FileNotFoundError(f"Required file not found: {src_path}")
                
    # 2. Write LICENSE
    license_path = os.path.join(hf_dir, "LICENSE")
    license_content = """Creative Commons Attribution 4.0 International (CC BY 4.0)
https://creativecommons.org/licenses/by/4.0/
"""
    with open(license_path, "w", encoding="utf-8") as f:
        f.write(license_content)
    print("  Created: LICENSE")
    
    # 3. Compute scenario scale statistics
    test_count = count_scenarios(os.path.join(repo_root, "data/retrace_bench/test_800_templateheldout_en/scenarios.jsonl"))
    calib_count = count_scenarios(os.path.join(repo_root, "data/retrace_bench/sample_80_hard_en/scenarios.jsonl"))
    
    if args.include_supervision:
        train_count = count_scenarios(os.path.join(repo_root, "data/retrace_supervision/train_3000_en/scenarios.jsonl"))
        dev_count = count_scenarios(os.path.join(repo_root, "data/retrace_supervision/dev_400_en/scenarios.jsonl"))
        total_count = test_count + calib_count + train_count + dev_count
        supervision_status_str = f"""- **supervision/train_3000_en**: {train_count} scenarios (contains SFT training targets)
- **supervision/dev_400_en**: {dev_count} scenarios"""
    else:
        total_count = test_count + calib_count
        supervision_status_str = "*(Supervision splits train_3000_en and dev_400_en were excluded from this packaging run)*"
        
    # 4. Generate README.md (Dataset Card)
    readme_path = os.path.join(hf_dir, "README.md")
    readme_content = f"""---
license: cc-by-4.0
language:
- en
pretty_name: ReTrace-Bench
task_categories:
- question-answering
- text-classification
- text-generation
tags:
- agent-memory
- llm-agents
- benchmark
- memory-revision
- long-term-memory
- reliability
configs:
- config_name: default
  data_files:
  - split: test
    path: benchmark/test_800_templateheldout_en/scenarios.jsonl
  - split: calibration
    path: calibration/sample_80_hard_en/scenarios.jsonl
  - split: train
    path: supervision/train_3000_en/scenarios.jsonl
  - split: dev
    path: supervision/dev_400_en/scenarios.jsonl
---

# ReTrace-Bench

ReTrace-Bench evaluates agent memory revision reliability in multi-agent and agentic workflows. It tests whether systems can correctly process new evidence to update, block, release, reaffirm, or reject memory states without introducing stale, out-of-scope, or policy-invalid memory.

## IMPORTANT Notice

- **benchmark/test_800_templateheldout_en** is the canonical, paper-facing held-out benchmark split.
- **Do not train, prompt-tune, policy-optimize, or select checkpoints on `benchmark/test_800_templateheldout_en`.**
- `calibration/sample_80_hard_en` is a small quickstart/calibration split designed for debugging and pipeline verification.
- `supervision/train_3000_en` and `supervision/dev_400_en` are synthetic supervision/selection pools for learning-based revision proposers. They are **NOT** benchmark tests and may contain `training_targets`.
- The old prototype/diagnostic split `test_800_en` is excluded from this public release package.

## Current Dataset Scale

- **benchmark/test_800_templateheldout_en**: {test_count} scenarios
- **calibration/sample_80_hard_en**: {calib_count} scenarios
{supervision_status_str}
- **Total packaged scenarios**: {total_count} scenarios

## Scenario Task Views

Each scenario evaluates an agent memory state through four distinct task views:

1. **`black_box_task`**: Evaluate end-to-end question answering utilizing revised memory state.
2. **`memory_state_task`**: Predict final eligibility statuses of all tracked beliefs.
3. **`evidence_retrieval_task`**: Identify evidence items that active memory changes are grounded upon.
4. **`diagnostic_task`**: Detect and classify memory-revision failures or conflicts.

For Hugging Face dataset viewer compatibility, nested scenario structures are
published as JSON string columns:

- `secondary_failure_modes_json`
- `public_input_json`
- `tasks_json`
- `hidden_gold_json`
- `metadata_json`
- `training_targets_json`

Parse these columns with `json.loads(...)` to recover the canonical nested
objects. The source-of-truth local files under `data/` keep the native nested
JSONL schema.

## Primary Metrics

Benchmark evaluations report the following primary metrics:

- `decision_macro_f1`
- `non_answer_decision_accuracy`
- `memory_state_accuracy`
- `evidence_f1`
- `failure_diagnosis_accuracy`
- `stale_reuse_rate`

## Quality Diagnostics for `test_800_templateheldout_en`

The held-out test split is designed with strict template-independent validation guarantees:
- Includes scenarios across **all 8 domains** and **all 11 failure modes**.
- **Template lookup coverage**: `0.000`
- **Template lookup decision accuracy**: `0.291`
- **Template lookup macro-F1**: `0.090`
- **Train/dev template signature overlap with candidate test**: `0.00%`

## Baseline Caveat

Please note that the **oracle** row documented in baseline results represents a diagnostic verification path for replaying typed state/evidence/diagnosis structures through the deterministic ReTrace-Engine. It is **not** a deployable memory baseline and should not be treated as a black-box decision upper bound.

## License

This dataset is distributed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE) license.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("  Created: README.md")
    
    print(f"\nSuccessfully packaged ReTrace-Bench dataset with {total_count} scenarios in total!")

if __name__ == "__main__":
    main()

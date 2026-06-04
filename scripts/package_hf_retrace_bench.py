#!/usr/bin/env python3
"""Package the ReTrace-Bench **v1.0** dataset for Hugging Face.

Copies the four paper-facing canonical splits from ``data/retrace_bench/`` into
``release/huggingface/ReTrace-Bench/`` using the public split names
``main`` / ``hard`` / ``realistic`` / ``calibration`` (never train / dev /
validation / test), and generates the dataset card.

The realistic split is annotation-pending: its empty annotation template ships
under ``annotations/`` and no fabricated human annotation is included.
"""

import argparse
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark.retrace_bench.public_view import sanitize_public_input

SCENARIO_JSONL_NAME = "scenarios.jsonl"

# Canonical public GitHub repository for the benchmark artifact. The dataset
# card links back here for the evaluator, schema, and reproducible baselines.
GITHUB_URL = "https://github.com/yuchenzhu-research/ReTrace"

# (on-disk split dir, public split name, HF jsonl path, expected count)
SPLITS = (
    ("main_3000_en", "main", "main/main_3000_en.jsonl", 3000),
    ("hard_500_en", "hard", "hard/hard_500_en.jsonl", 500),
    ("realistic_200_en", "realistic", "realistic/realistic_200_en.jsonl", 200),
    ("calibration_80_en", "calibration", "calibration/calibration_80_en.jsonl", 80),
)


def get_benchmark_version(repo_root):
    """Best-effort read of benchmark.retrace_bench.__version__ without importing."""
    init_path = os.path.join(repo_root, "benchmark", "retrace_bench", "__init__.py")
    try:
        text = open(init_path, encoding="utf-8").read()
    except OSError:
        return "unknown"
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    return match.group(1) if match else "unknown"


def to_viewer_row(scenario):
    """Flatten nested scenario payloads so the Hugging Face viewer can infer a stable schema."""
    row = {
        "scenario_id": scenario["scenario_id"],
        "split": scenario.get("split", ""),
        "domain": scenario["domain"],
        "primary_failure_mode": scenario["primary_failure_mode"],
        "secondary_failure_modes_json": json.dumps(
            scenario.get("secondary_failure_modes", []), ensure_ascii=False, sort_keys=True
        ),
        "difficulty": scenario["difficulty"],
        "workflow_context": scenario["workflow_context"],
        "public_input_json": json.dumps(
            sanitize_public_input(scenario.get("public_input", {})),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "tasks_json": json.dumps(scenario.get("tasks", []), ensure_ascii=False, sort_keys=True),
        "hidden_gold_json": json.dumps(
            scenario.get("hidden_gold", {}), ensure_ascii=False, sort_keys=True
        ),
        "metadata_json": json.dumps(
            scenario.get("metadata", {}), ensure_ascii=False, sort_keys=True
        ),
    }
    return row


def copy_jsonl_for_viewer(src_path, tgt_path):
    with open(src_path, "r", encoding="utf-8") as src, open(tgt_path, "w", encoding="utf-8") as tgt:
        for line in src:
            if not line.strip():
                continue
            tgt.write(json.dumps(to_viewer_row(json.loads(line)), ensure_ascii=False) + "\n")


def count_scenarios(jsonl_path):
    if not os.path.exists(jsonl_path):
        return 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def parse_args():
    parser = argparse.ArgumentParser(description="Package ReTrace-Bench v1.0 dataset for Hugging Face.")
    return parser.parse_args()


def build_readme(counts, benchmark_version, configs_block):
    main_c, hard_c, real_c, calib_c = (
        counts["main"], counts["hard"], counts["realistic"], counts["calibration"]
    )
    total = main_c + hard_c + real_c + calib_c
    return f"""---
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
- long-context
- reliability
- evaluation
{configs_block}
---

# ReTrace-Bench

ReTrace-Bench v{benchmark_version} evaluates **agent memory revision
reliability**: whether a system can process new evidence to update, block,
release, reaffirm, or reject memory states without introducing stale,
out-of-scope, or policy-invalid memory. It is not merely a final-answer
benchmark — coarse decision accuracy can overestimate memory reliability, so the
benchmark also scores memory-state tracking, evidence grounding, and failure
diagnosis.

## 1. Dataset Summary

Four paper-facing splits, all English, controlled or realistic-style synthetic,
constructed with a leakage-audited (de-actionalized) procedure: authoritative
records never contain a decision-action phrase, so the correct revision decision
must be recovered by reasoning over described state rather than string matching.

## 2. Split Overview

| split | size | role |
|-------|------|------|
| `main` | {main_c} | controlled benchmark main split |
| `hard` | {hard_c} | long-context and multi-evidence stress split |
| `realistic` | {real_c} | realistic-style workflow split, annotation pending |
| `calibration` | {calib_c} | smoke/quickstart only |

## 3. Task Definition

Each scenario presents an initial memory set and a chronological event trace.
The system must decide how memory should be revised and answer four task views:
black-box answer, memory-state classification, evidence retrieval, and failure
diagnosis.

## 4. Scenario Schema

Source-of-truth scenarios are nested JSON objects with `scenario_id`, `split`,
`domain`, `primary_failure_mode`, `difficulty`, `workflow_context`,
`public_input` (`initial_memory`, `event_trace`), `tasks`, `hidden_gold`, and
`metadata`. So the Hugging Face viewer can render every column, nested fields are published as
JSON string columns (`public_input_json`, `tasks_json`, `hidden_gold_json`,
`metadata_json`, `secondary_failure_modes_json`); parse them with
`json.loads(...)`.

## 5. Prediction Schema

One JSON object per line, matched to scenarios by `scenario_id`:

```json
{{
  "scenario_id": "<scenario id>",
  "response": {{
    "answer": "<free-text answer>",
    "decision": "use_current_memory",
    "memory_state": {{"<memory_id>": "outdated"}},
    "evidence_event_ids": ["<event_id from public_input.event_trace>"],
    "failure_diagnosis": "stale_memory_reuse"
  }}
}}
```

- `decision`: one of `use_current_memory`, `escalate`, `ask_clarification`,
  `refuse_due_to_policy`, `mark_unresolved`.
- `memory_state`: `memory_id -> status` (`current`, `outdated`, `blocked`,
  `unresolved`, `out_of_scope`, `deleted`, `should_not_store`, `restored`).
- `evidence_event_ids`: `event_id` values from `public_input.event_trace`.
- `failure_diagnosis`: one of the eleven failure-mode labels.

## 6. Official Evaluator

ReTrace-Bench ships an official scorer that runs no model and needs no API keys.
Clone the repository at {GITHUB_URL}, then score a predictions file:

```bash
PYTHONPATH=. python scripts/evaluate_retrace_bench_predictions.py \\
  --data data/retrace_bench/main_3000_en/scenarios.jsonl \\
  --predictions path/to/predictions.jsonl \\
  --out-metrics outputs/retrace_bench/my_model.metrics.json \\
  --out-scored outputs/retrace_bench/my_model.scored.jsonl \\
  --print-table
```

See `examples/retrace_bench/` for a runnable example and the Python API
(`benchmark.retrace_bench.api`).

## 7. Metrics

Primary metrics: `decision_macro_f1`, `non_answer_decision_accuracy`,
`memory_state_accuracy`, `evidence_f1`, `failure_diagnosis_accuracy`,
`stale_reuse_rate`.

## 8. Benchmark Hygiene / Leakage Audit

Every split passes a decision-word leakage audit: no verified/trusted
(authoritative) record contains a decision-action phrase tied to one of the five
gold decisions. Scenario, memory, and event IDs are disjoint across splits, and
there is no universal cross-scope distractor shortcut.

## 9. Annotation Status

- `main`, `hard`, `calibration`: `controlled_synthetic`, synthetic gold.
- `realistic`: `realistic_style_synthetic`, **`annotation_status = pending`**
  (synthetic gold, **not** human-reviewed). It is **never** auto-marked
  `reviewed`; a `reviewed` status is only valid once real human annotators have
  completed the validation protocol. No fabricated human annotation is included
  in this release.

## 10. Intended Use

`main` is for primary benchmark results; `hard` for long-context / multi-evidence
stress; `realistic` for realistic workflow texture once annotated. `calibration`
is a smoke/quickstart split only: **it is not a model-selection / checkpoint-selection validation set and must not be used to tune or select systems**, and it must not be used for headline claims.

## 11. Limitations

`main` / `hard` / `calibration` gold is synthetic. `realistic` is unannotated in
this release. The legacy pre-v1.0 layout is not part of this release and is
recoverable only from the Git tag `legacy-retrace-bench-pre-v1.0`.

## 12. License

Distributed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE) license.

*Total packaged scenarios: {total}.*
"""


def main():
    parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    hf_dir = os.path.join(repo_root, "release", "huggingface", "ReTrace-Bench")

    if os.path.exists(hf_dir):
        print(f"Cleaning existing Hugging Face package directory at {hf_dir}...")
        shutil.rmtree(hf_dir)
    os.makedirs(hf_dir, exist_ok=True)

    counts = {}
    for dir_name, public, hf_path, expected in SPLITS:
        src = os.path.join(repo_root, "data", "retrace_bench", dir_name, SCENARIO_JSONL_NAME)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Required split not found: {src}")
        tgt = os.path.join(hf_dir, hf_path)
        os.makedirs(os.path.dirname(tgt), exist_ok=True)
        copy_jsonl_for_viewer(src, tgt)
        n = count_scenarios(src)
        counts[public] = n
        if n != expected:
            raise ValueError(f"{public} split has {n} scenarios, expected {expected}")
        print(f"  Packaged {public}: {src} -> {hf_path} ({n} scenarios)")



    with open(os.path.join(hf_dir, "LICENSE"), "w", encoding="utf-8") as f:
        f.write(
            "Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
            "https://creativecommons.org/licenses/by/4.0/\n"
        )
    print("  Created: LICENSE")

    config_lines = ["configs:", "- config_name: default", "  data_files:"]
    for _, public, hf_path, _ in SPLITS:
        config_lines.append(f"  - split: {public}")
        config_lines.append(f"    path: {hf_path}")
    configs_block = "\n".join(config_lines)

    readme = build_readme(
        counts, get_benchmark_version(repo_root), configs_block
    )
    with open(os.path.join(hf_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    print("  Created: README.md")

    total = sum(counts.values())
    print(f"\nSuccessfully packaged ReTrace-Bench v1.0 with {total} scenarios across 4 splits!")


if __name__ == "__main__":
    main()

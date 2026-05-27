from __future__ import annotations

import json
import os
from typing import Any


def load_records(file_path: str) -> list[dict[str, Any]]:
    if not os.path.exists(file_path):
        return []
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def generate_ablation_table(output_dir: str = "outputs") -> str:
    """Loads outputs and generates a markdown table summarizing ablations and costs."""
    methods = [
        ("retrieval_baseline", "Retrieval Baseline"),
        ("retrace", "ReTrace (Full)"),
        ("retrace_no_ledger", "ReTrace w/o Ledger"),
        ("retrace_no_gate", "ReTrace w/o TMS Gate"),
        ("retrace_no_temporal", "ReTrace w/o Temporal"),
    ]

    lines = []
    lines.append("| Method | STALE Rec. Count | Memora Rec. Count | Avg. Prompt Tokens | Avg. Completion Tokens |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for method_key, method_name in methods:
        stale_path = os.path.join(output_dir, "stale", f"{method_key}_frozen_eval.jsonl")
        memora_path = os.path.join(output_dir, "memora", f"{method_key}_frozen_eval.jsonl")

        stale_recs = load_records(stale_path)
        memora_recs = load_records(memora_path)

        # Calculate average tokens if tokens info exists in records
        total_prompt = 0
        total_completion = 0
        total_calls = 0
        count = 0

        for r in stale_recs + memora_recs:
            tokens = r.get("tokens", {})
            if tokens:
                total_prompt += tokens.get("prompt_tokens", 0)
                total_completion += tokens.get("completion_tokens", 0)
                count += 1

        avg_prompt = f"{total_prompt / count:.1f}" if count > 0 else "N/A"
        avg_completion = f"{total_completion / count:.1f}" if count > 0 else "N/A"

        lines.append(
            f"| {method_name} | {len(stale_recs)} | {len(memora_recs)} | {avg_prompt} | {avg_completion} |"
        )

    table_str = "\n".join(lines)
    return table_str


if __name__ == "__main__":
    # Print the table locally
    print("==================================================")
    print("Ablation & Cost Summary Table (Table 4)")
    print("==================================================")
    print(generate_ablation_table())
    print("==================================================")

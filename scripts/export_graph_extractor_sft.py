#!/usr/bin/env python3
"""Export graph extractor SFT training dataset in ShareGPT / Messages format.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export graph extractor SFT training data.")
    parser.add_argument("--in-file", type=str, default="outputs/raw_dialogue_synth.jsonl", help="Input dataset path.")
    parser.add_argument("--out-file", type=str, default="outputs/graph_extractor_sft.jsonl", help="Output SFT dataset path.")
    parser.add_argument("--dry-run", action="store_true", help="Print sample pair and do not write to file.")
    args = parser.parse_args()

    if not os.path.exists(args.in_file):
        print(f"Error: input file {args.in_file} does not exist. Run scripts/build_raw_dialogue_synth.py first.")
        sys.exit(1)

    sft_rows = []
    with open(args.in_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            system_prompt = (
                "You are the ReTrace Graph Extractor. Your task is to extract a structured memory graph "
                "from raw subagent dialogue.\n"
                "Return ONLY a strict JSON object with these keys:\n"
                "1. 'evidence_nodes'\n"
                "2. 'belief_nodes'\n"
                "3. 'condition_nodes'\n"
                "4. 'candidate_replacement_beliefs'\n"
                "5. 'dependency_edges'\n"
                "Do NOT output markdown code blocks or extra explanations."
            )

            # Reconstruct raw dialogue string
            dialogue_parts = []
            for u in data["dialogue"]["utterances"]:
                dialogue_parts.append(f"{u['speaker']}: {u['text']}")
            dialogue_str = "\n".join(dialogue_parts)

            user_content = (
                f"Subagent Roles: {', '.join(data['subagent_roles'])}\n\n"
                f"Raw Dialogue:\n{dialogue_str}"
            )

            assistant_content = json.dumps(data["gold_graph"])

            row = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            }
            sft_rows.append(row)

    if args.dry_run:
        print("=== DRY RUN SAMPLE ===")
        print(json.dumps(sft_rows[0], indent=2))
        return

    # Ensure out dir exists
    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_file, "w", encoding="utf-8") as f:
        for r in sft_rows:
            f.write(json.dumps(r) + "\n")

    print(f"Successfully exported {len(sft_rows)} SFT examples to {args.out_file}.")


if __name__ == "__main__":
    main()

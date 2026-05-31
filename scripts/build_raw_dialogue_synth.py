#!/usr/bin/env python3
"""CLI script to build a synthetic dataset of subagent dialogues and labeled memory graphs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from retracemem.evaluation.raw_dialogue.generator import SyntheticDialogueGenerator
from retracemem.evaluation.raw_dialogue.contracts import DialogueExtractionTarget, RawDialogue


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic raw dialogues.")
    parser.add_argument("--out", type=str, default="outputs/raw_dialogue_synth.jsonl", help="Output JSONL file path.")
    parser.add_argument("--num-examples", "-n", "--n", type=int, default=10, dest="num_examples", help="Number of examples to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation.")
    args = parser.parse_args()

    # Ensure output dir exists
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    generator = SyntheticDialogueGenerator(seed=args.seed)

    print(f"Generating {args.num_examples} examples with seed {args.seed}...")
    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for idx in range(args.num_examples):
            ex_id = f"synth_ex_{idx}"
            family = generator.case_families[idx % len(generator.case_families)]
            data = generator.generate_episode(ex_id, case_family=family)

            # Convert to target contract
            raw_dialogue_lines = data["raw_dialogue"].strip().split("\n")
            utterances = []
            for line in raw_dialogue_lines:
                if ":" in line:
                    speaker, text = line.split(":", 1)
                    speaker = speaker.strip()
                    text = text.strip()
                else:
                    speaker = "unknown"
                    text = line.strip()
                utterances.append({
                    "speaker": speaker,
                    "text": text
                })

            target_dict = {
                "example_id": ex_id,
                "dialogue": {
                    "utterances": utterances
                },
                "subagent_roles": data["subagent_roles"],
                "gold_graph": data["gold_graph"],
                "metadata": {
                    "new_evidence_id": data["new_evidence_id"],
                    "gold_actions": data["gold_actions"],
                    "gold_final_statuses": data["gold_final_statuses"]
                }
            }

            # Validate before writing
            target = DialogueExtractionTarget.from_dict(target_dict)
            target.validate()

            f.write(json.dumps(target.to_dict()) + "\n")
            written += 1

    print(f"Successfully generated and validated {written} examples at {args.out}.")


if __name__ == "__main__":
    main()

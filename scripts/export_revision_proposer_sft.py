#!/usr/bin/env python3
"""Export revision proposer SFT training dataset in ShareGPT / Messages format.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export revision proposer SFT training data.")
    parser.add_argument("--in-file", type=str, default="outputs/raw_dialogue_synth.jsonl", help="Input dataset path.")
    parser.add_argument("--out-file", type=str, default="outputs/typed_revision_sft.jsonl", help="Output SFT dataset path.")
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
                "You are the ReTrace Stage C revision policy. Your task is to propose explicit "
                "typed revision proposals for multi-agent shared-memory updates.\n\n"
                "Propose revision actions only from this canonical vocabulary:\n"
                "- SUPERSEDES (requires target_belief_id and replacement_belief_id)\n"
                "- BLOCKS (requires target_condition_id)\n"
                "- RELEASES (requires target_condition_id)\n"
                "- UNCERTAIN (requires target_belief_id)\n"
                "- REAFFIRMS (requires target_belief_id)\n"
                "- NO_REVISION (requires all target/replacement IDs as null)\n\n"
                "Do not output any thinking process or markdown formatting. Return ONLY a strict JSON array of objects."
            )

            g = data["gold_graph"]
            new_ev_id = data["metadata"]["new_evidence_id"]
            new_ev_node = next((e for e in g["evidence_nodes"] if e["evidence_id"] == new_ev_id), None)
            new_evidence_text = f'"{new_ev_node["text"]}" (ID: {new_ev_id})' if new_ev_node else f"(ID: {new_ev_id})"

            beliefs_str = "\n".join(f"  - {b['belief_id']}: \"{b['proposition']}\"" for b in g["belief_nodes"]) or "  - (none)"
            replacements_str = "\n".join(f"  - {b['belief_id']}: \"{b['proposition']}\"" for b in g["candidate_replacement_beliefs"]) or "  - (none)"

            cond_parts = []
            # We map condition nodes by belief based on dependency edges
            for edge in g["dependency_edges"]:
                bid = edge["belief_id"]
                cid = edge["condition_id"]
                cond = next((c for c in g["condition_nodes"] if c["condition_id"] == cid), None)
                if cond:
                    cond_parts.append(f"  - [{bid}] {cid}: \"{cond['text']}\"")
            conditions_str = "\n".join(cond_parts) or "  - (none)"

            user_content = (
                "### Deterministic Candidate Contrast Block\n"
                f"- New Evidence to Evaluate: {new_evidence_text}\n\n"
                "- Prior Candidate Beliefs:\n"
                f"{beliefs_str}\n\n"
                "- Candidate Replacement Beliefs:\n"
                f"{replacements_str}\n\n"
                "- Condition Anchors:\n"
                f"{conditions_str}"
            )

            assistant_content = json.dumps(data["metadata"]["gold_actions"])

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

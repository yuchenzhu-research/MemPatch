from __future__ import annotations

import json
import os
import sys

# Ensure src is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "src"))

from retracemem.evaluation.multiagent.data.stagec_dataset import build_stagec_dataset
from retracemem.proposers.typed_revision_policy import PromptTypedRevisionPolicy


def main() -> None:
    print("[*] Preparing MLX-LM SFT training data...")
    examples = build_stagec_dataset()
    policy = PromptTypedRevisionPolicy()
    
    records = []
    for ex in examples:
        # Build prompt using policy rules
        messages = list(policy.build_messages(ex.method_visible_input))
        
        # Serialize gold targets to assistant message
        targets_list = []
        for t in ex.targets:
            targets_list.append({
                "action_type": t.action_type,
                "target_belief_id": t.target_belief_id,
                "target_condition_id": t.target_condition_id,
                "replacement_belief_id": t.replacement_belief_id,
                "rationale": t.rationale or "",
                "evidence_ids": list(t.evidence_ids),
            })
        assistant_content = json.dumps(targets_list, ensure_ascii=False)
        
        # MLX-LM format: messages chat style
        records.append({
            "messages": [
                {"role": "system", "content": messages[0]["content"]},
                {"role": "user", "content": messages[1]["content"]},
                {"role": "assistant", "content": assistant_content}
            ]
        })
        
    # Split into train/validation sets (e.g. 18 train, 6 valid)
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, "train.jsonl")
    valid_path = os.path.join(data_dir, "valid.jsonl")
    
    with open(train_path, "w", encoding="utf-8") as f:
        for rec in records[:18]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    with open(valid_path, "w", encoding="utf-8") as f:
        for rec in records[18:]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print(f"[+] Data prepared in: {data_dir}/")
    print(f"    - Train count: {len(records[:18])}")
    print(f"    - Valid count: {len(records[18:])}")


if __name__ == "__main__":
    main()

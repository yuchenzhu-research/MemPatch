from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any, Dict, List
from experiments.multiagent.stagec_dataset import build_stagec_dataset
from experiments.multiagent.contracts import StageCTrainingExample

SYSTEM_PROMPT = (
    "You are the ReTrace Stage C revision policy. Your task is to propose explicit "
    "typed revision proposals for multi-agent shared-memory updates. "
    "Propose revision actions only from this canonical vocabulary:\n"
    "- SUPERSEDES (requires replacement_belief_id)\n"
    "- BLOCKS\n"
    "- RELEASES\n"
    "- UNCERTAIN\n"
    "- REAFFIRMS\n"
    "- NO_REVISION\n\n"
    "Return your response as a strict JSON array of objects with the following fields:\n"
    "- action_type (string)\n"
    "- target_belief_id (string or null)\n"
    "- target_condition_id (string or null)\n"
    "- replacement_belief_id (string or null)\n"
    "- rationale (string)\n"
    "- evidence_ids (array of strings)\n\n"
    "Example format:\n"
    '[\n  {\n    "action_type": "SUPERSEDES",\n    "target_belief_id": "b_old",\n'
    '    "replacement_belief_id": "b_new",\n    "rationale": "...",\n'
    '    "evidence_ids": ["ev_1"]\n  }\n]'
)


def format_user_prompt(ex: StageCTrainingExample) -> str:
    """Format the method-visible context of a submission for the user message."""
    sub = ex.method_visible_input
    
    # 1. Candidate Beliefs
    candidates = []
    for b in sub.candidate_beliefs:
        candidates.append(f"- ID: {b.belief_id}, Proposition: {b.proposition}")
        
    # 2. Candidate Replacement Beliefs
    replacements = []
    for b in sub.candidate_replacement_beliefs:
        replacements.append(f"- ID: {b.belief_id}, Proposition: {b.proposition}")
        
    # 3. Evidence Context
    evidence = []
    for ev in sub.evidence_context:
        evidence.append(f"- ID: {ev.evidence_id}, Content: {ev.text}")
        
    prompt = (
        f"Episode ID: {ex.episode_id}\n"
        f"Submission ID: {sub.submission_id}\n"
        f"Domain: {ex.domain}\n"
        f"Query: {sub.query}\n\n"
        "Candidate Beliefs:\n" + ("\n".join(candidates) if candidates else "None") + "\n\n"
        "Candidate Replacement Beliefs:\n" + ("\n".join(replacements) if replacements else "None") + "\n\n"
        "Evidence Context:\n" + ("\n".join(evidence) if evidence else "None") + "\n\n"
        f"New Evidence ID: {sub.new_evidence_id}\n"
        "Identify the correct revision actions."
    )
    return prompt


def format_assistant_response(ex: StageCTrainingExample) -> str:
    """Format targets into a strict JSON string."""
    targets_dict = []
    for t in ex.targets:
        if t.action_type == "NO_REVISION":
            continue
        targets_dict.append({
            "action_type": t.action_type,
            "target_belief_id": t.target_belief_id,
            "target_condition_id": t.target_condition_id,
            "replacement_belief_id": t.replacement_belief_id,
            "rationale": t.rationale,
            "evidence_ids": list(t.evidence_ids),
        })
    return json.dumps(targets_dict, indent=2)


def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def export_stagec_sft() -> Dict[str, Any]:
    examples = build_stagec_dataset()
    rows = []
    
    for ex in examples:
        user_content = format_user_prompt(ex)
        assistant_content = format_assistant_response(ex)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        
        row = {
            "example_id": ex.example_id,
            "episode_id": ex.episode_id,
            "submission_id": ex.submission_id,
            "split": "development_only",
            "domain": ex.domain,
            "failure_type": ex.failure_type,
            "label_source": "human_authored_seed",
            "training_eligible": False,
            "messages": messages,
            "metadata": {
                "scientific_status": "pipeline_validation_only",
                "contains_gold_in_user_input": False,
            }
        }
        rows.append(row)
        
    os.makedirs("outputs", exist_ok=True)
    jsonl_path = "outputs/stagec_dev_preview_train.jsonl"
    
    # Write jsonl
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            
    # Compute hashes for reproducibility manifest
    # Prompt template hash
    prompt_hash = compute_sha256(SYSTEM_PROMPT)
    
    # Exported SFT data hash
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data_hash = compute_sha256(f.read())
        
    # Episode factory hash (read source dataset module code)
    factory_code = ""
    factory_path = "experiments/multiagent/stagec_dataset.py"
    if os.path.exists(factory_path):
        with open(factory_path, "r", encoding="utf-8") as f:
            factory_code = f.read()
    factory_hash = compute_sha256(factory_code) if factory_code else "placeholder_hash"
    
    manifest = {
        "dataset_name": "retrace_stagec_dev_preview",
        "split": "development_only",
        "example_count": len(rows),
        "prompt_template_hash": prompt_hash,
        "exported_dataset_hash": data_hash,
        "episode_factory_hash": factory_hash,
        "notes": "SFT dev-preview exported from 14 development seeds.",
    }
    
    manifest_path = "outputs/stagec_dev_preview_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    return {
        "example_count": len(rows),
        "jsonl_path": jsonl_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Stage C SFT preview data (Packet 4D)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dev_preview",
        choices=["dev_preview"],
        help="Export mode",
    )
    args = parser.parse_args()
    
    res = export_stagec_sft()
    print("Stage C SFT Preview exported successfully.")
    print(f"File: {res['jsonl_path']}, Manifest: {res['manifest_path']}")
    print(f"Total SFT examples: {res['example_count']}")
    print(f"Exported SFT dataset SHA256: {res['manifest']['exported_dataset_hash']}")


if __name__ == "__main__":
    main()

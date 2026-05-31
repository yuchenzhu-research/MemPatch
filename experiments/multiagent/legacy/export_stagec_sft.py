from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any, Dict, List
from retracemem.evaluation.multiagent.data.stagec_dataset import build_stagec_dataset
from retracemem.evaluation.multiagent.contracts import StageCTrainingExample

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
    "Constraints:\n"
    "- BLOCKS and RELEASES must target only listed condition IDs (target_condition_id).\n"
    "- SUPERSEDES, UNCERTAIN, and REAFFIRMS must target only listed belief IDs (target_belief_id).\n"
    "- NO_REVISION must specify target_belief_id, target_condition_id, and replacement_belief_id as null, and include the new_evidence_id in evidence_ids.\n\n"
    "Return your response as a strict JSON array of objects with the following fields:\n"
    "- action_type (string)\n"
    "- target_belief_id (string or null)\n"
    "- target_condition_id (string or null)\n"
    "- replacement_belief_id (string or null)\n"
    "- rationale (string)\n"
    "- evidence_ids (array of strings)\n\n"
    "Example format for NO_REVISION:\n"
    '[\n  {\n    "action_type": "NO_REVISION",\n    "target_belief_id": null,\n'
    '    "target_condition_id": null,\n    "replacement_belief_id": null,\n'
    '    "rationale": "No evidence-grounded revision is warranted.",\n'
    '    "evidence_ids": ["ev_new"]\n  }\n]'
)


def format_user_prompt(ex: StageCTrainingExample) -> str:
    """Format the method-visible context of a submission for the user message."""
    sub = ex.method_visible_input
    
    # 1. Submission Metadata
    meta_lines = [
        f"- Submission ID: {sub.submission_id}",
        f"- Producer ID: {sub.producer_id}",
        f"- Producer Role: {sub.producer_role}",
        f"- Observed At: {sub.observed_at}",
        f"- Parent Snapshot ID: {sub.parent_snapshot_id}",
    ]
    if sub.task_id:
        meta_lines.append(f"- Task ID: {sub.task_id}")
        
    # 2. Evidence Context
    evidence_lines = []
    for ev in sub.evidence_context:
        source_ptr_str = f", Source: {ev.source_pointer}" if hasattr(ev, "source_pointer") and ev.source_pointer else ""
        timestamp_str = f", Timestamp: {ev.timestamp}" if hasattr(ev, "timestamp") and ev.timestamp else ""
        evidence_lines.append(f"- ID: {ev.evidence_id}{timestamp_str}{source_ptr_str}, Content: {ev.text}")
        
    # 3. Candidate Beliefs
    candidates = []
    for b in sub.candidate_beliefs:
        ev_ids_str = ", ".join(b.source_evidence_ids) if b.source_evidence_ids else "None"
        candidates.append(f"- ID: {b.belief_id}, Proposition: {b.proposition}, Source Evidence: [{ev_ids_str}]")
        
    # 4. Candidate Replacement Beliefs
    replacements = []
    for b in sub.candidate_replacement_beliefs:
        ev_ids_str = ", ".join(b.source_evidence_ids) if b.source_evidence_ids else "None"
        replacements.append(f"- ID: {b.belief_id}, Proposition: {b.proposition}, Source Evidence: [{ev_ids_str}]")
        
    # 5. Candidate Conditions by Belief
    conditions = []
    for bid, conds in sub.candidate_conditions_by_belief:
        for c in conds:
            conditions.append(f"- Owning Belief: {bid}, Condition ID: {c.condition_id}, Scope: {c.scope_id}, Text: {c.text}")
            
    # 6. Pre-existing Dependency Anchors
    dependencies = []
    for bid, deps in sub.dependency_edges_by_belief:
        for d in deps:
            dependencies.append(f"- {d.belief_id} --REQUIRES--> {d.condition_id}")
            
    prompt = (
        f"Episode ID: {ex.episode_id}\n\n"
        "Submission Metadata:\n" + "\n".join(meta_lines) + "\n\n"
        f"Query: {sub.query}\n\n"
        "Evidence Context:\n" + ("\n".join(evidence_lines) if evidence_lines else "None") + "\n\n"
        f"New Evidence ID: {sub.new_evidence_id}\n\n"
        "Candidate Beliefs:\n" + ("\n".join(candidates) if candidates else "None") + "\n\n"
        "Candidate Replacement Beliefs:\n" + ("\n".join(replacements) if replacements else "None") + "\n\n"
        "Candidate Conditions by Belief:\n" + ("\n".join(conditions) if conditions else "None") + "\n\n"
        "Pre-existing Dependency Anchors:\n" + ("\n".join(dependencies) if dependencies else "None") + "\n\n"
        "Identify the correct revision actions. Return a strict JSON array of objects."
    )
    return prompt


def format_assistant_response(ex: StageCTrainingExample) -> str:
    """Format targets into a strict JSON string."""
    targets_dict = []
    new_evidence_id = ex.method_visible_input.new_evidence_id
    for t in ex.targets:
        if t.action_type == "NO_REVISION":
            ev_ids = list(t.evidence_ids) if t.evidence_ids else [new_evidence_id]
            targets_dict.append({
                "action_type": "NO_REVISION",
                "target_belief_id": None,
                "target_condition_id": None,
                "replacement_belief_id": None,
                "rationale": t.rationale or "No evidence-grounded revision is warranted.",
                "evidence_ids": ev_ids,
            })
        else:
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

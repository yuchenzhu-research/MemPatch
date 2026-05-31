from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from typing import Any, Dict, List
from retracemem.evaluation.multiagent.data.dev_expansion import generate_expanded_episodes

def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def export_review_queue() -> Dict[str, Any]:
    episodes_with_gold = generate_expanded_episodes()
    
    rows: List[Dict[str, Any]] = []
    
    for ep, gold in episodes_with_gold:
        # Construct summary of method-visible input
        visible_subs = []
        for s in ep.submissions:
            b_ids = [b.belief_id for b in s.candidate_beliefs]
            visible_subs.append(f"Sub {s.submission_id} (New ev: {s.new_evidence_id}, Beliefs: {b_ids})")
        input_summary = "; ".join(visible_subs)
        
        # Construct summary of proposed gold target
        target_summaries = []
        for t in gold.gold_typed_targets:
            if t.action_type == "NO_REVISION":
                target_summaries.append("NO_REVISION")
            elif t.action_type == "SUPERSEDES":
                target_summaries.append(f"SUPERSEDES {t.target_belief_id} with {t.replacement_belief_id}")
            elif t.action_type in ("BLOCKS", "RELEASES"):
                target_summaries.append(f"{t.action_type} {t.target_condition_id}")
            else:
                target_summaries.append(f"{t.action_type} {t.target_belief_id}")
        target_summary = ", ".join(target_summaries) if target_summaries else "None"
        
        row = {
            "episode_id": ep.episode_id,
            "domain": ep.domain,
            "failure_type": ep.failure_type_public_or_controlled,
            "subagent_roles": list(ep.subagent_roles),
            "number_of_submissions": len(ep.submissions),
            "downstream_task": ep.downstream_tasks[0].query if ep.downstream_tasks else "None",
            "method-visible input summary": input_summary,
            "proposed typed target summary": target_summary,
            "review_status": "pending_human_review",
            "reviewer_notes": "",
            "approve_for_prompt_smoke": False,
            "approve_for_training": False,
            "semantic_checklist": ep.metadata.get("semantic_checklist", {}),
            "semantic_validation_status": ep.metadata.get("semantic_validation_status", {}),
        }
        rows.append(row)
        
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Export CSV
    csv_path = "outputs/stagec_dev_review_sheet_70.csv"
    fields = [
        "episode_id", "domain", "failure_type", "subagent_roles", "number_of_submissions",
        "downstream_task", "method-visible input summary", "proposed typed target summary",
        "review_status", "reviewer_notes", "approve_for_prompt_smoke", "approve_for_training",
        "semantic_checklist", "semantic_validation_status"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            r_copy = r.copy()
            r_copy["subagent_roles"] = str(r_copy["subagent_roles"])
            r_copy["semantic_checklist"] = str(r_copy["semantic_checklist"])
            r_copy["semantic_validation_status"] = str(r_copy["semantic_validation_status"])
            writer.writerow(r_copy)
            
    # 2. Export JSONL
    jsonl_path = "outputs/stagec_dev_review_queue_70.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            
    # Compute checksums
    with open(csv_path, "r", encoding="utf-8") as f:
        csv_hash = compute_sha256(f.read())
    with open(jsonl_path, "r", encoding="utf-8") as f:
        jsonl_hash = compute_sha256(f.read())
        
    # 3. Export Manifest
    manifest = {
        "dataset_name": "stagec_dev_review_queue_70",
        "split": "development_candidate",
        "record_count": len(rows),
        "csv_path": csv_path,
        "jsonl_path": jsonl_path,
        "csv_sha256": csv_hash,
        "jsonl_sha256": jsonl_hash,
        "review_status": "pending_human_review",
        "notes": "Expanded 70-example dev review queue before manual labeling/approval."
    }
    
    manifest_path = "outputs/stagec_dev_review_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    return {
        "record_count": len(rows),
        "csv_path": csv_path,
        "jsonl_path": jsonl_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Stage C 70-Example Dev Review Queue (Packet 4E)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="development_candidate",
        choices=["development_candidate"],
        help="Export mode",
    )
    args = parser.parse_args()
    
    res = export_review_queue()
    print("Stage C Dev Review Queue exported successfully.")
    print(f"CSV: {res['csv_path']}, Hash: {res['manifest']['csv_sha256']}")
    print(f"JSONL: {res['jsonl_path']}, Hash: {res['manifest']['jsonl_sha256']}")
    print(f"Manifest: {res['manifest_path']}")
    print(f"Total review records: {res['record_count']}")

if __name__ == "__main__":
    main()

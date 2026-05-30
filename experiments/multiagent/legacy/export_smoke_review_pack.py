from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Tuple
from experiments.multiagent.dev_expansion import generate_expanded_episodes
from experiments.multiagent.export_stagec_sft import format_user_prompt
from experiments.multiagent.contracts import StageCTrainingExample
from experiments.multiagent.validate_candidate_semantics import validate_episode

def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def export_smoke_review_pack() -> Dict[str, Any]:
    episodes_with_gold = generate_expanded_episodes()
    
    # Enforce executable validation
    print("[*] Performing executable semantic consistency validation on all candidates before exporting review pack...")
    for ep, gold in episodes_with_gold:
        is_pass, detail = validate_episode(ep, gold)
        if not is_pass:
            raise ValueError(
                f"Candidate consistency error: Episode '{ep.episode_id}' is inconsistent with expected gold statuses. "
                f"Mismatches: {detail['mismatches']}"
            )
    print("[+] All candidates successfully passed executable semantic validation!")
    
    # 7 failure types in strict order, alternating domains
    selected_targets = [
        ("software_engineering", "direct_supersession"),
        ("research_workflow", "stale_propagation"),
        ("software_engineering", "scope_expansion"),
        ("research_workflow", "cross_agent_conflict"),
        ("software_engineering", "temporary_blocker_recovery"),
        ("research_workflow", "duplicate_evidence"),
        ("software_engineering", "ambiguous_update"),
    ]
    
    selected_pairs = []
    for domain, f_type in selected_targets:
        # Pick variant 1
        for ep, gold in episodes_with_gold:
            if ep.domain == domain and ep.failure_type_public_or_controlled == f_type and ep.episode_id.endswith("_v1"):
                selected_pairs.append((ep, gold))
                break
                
    # 1. Export JSONL and MD Review Pack
    md_lines = [
        "# ReTrace Stage C Seven-Case Prompt-Smoke Approval Pack",
        "",
        "This document contains 7 candidate episodes selected for human review before executing the first live Stage C prompt smoke.",
        "Please review each episode's method-visible inputs, expected targets, and semantic checklist.",
        "",
    ]
    
    jsonl_rows = []
    
    for idx, (ep, gold) in enumerate(selected_pairs, 1):
        last_sub = ep.submissions[-1]
        
        # Build user prompt for Markdown visualization
        # We need a StageCTrainingExample wrapper to format the prompt
        fake_ex = StageCTrainingExample(
            example_id=f"ex_{ep.episode_id}_{last_sub.submission_id}",
            episode_id=ep.episode_id,
            submission_id=last_sub.submission_id,
            method_visible_input=last_sub,
            targets=gold.gold_typed_targets,
            split=ep.split,
            domain=ep.domain,
            failure_type=ep.failure_type_public_or_controlled,
            label_source="human_authored",
        )
        user_prompt_text = format_user_prompt(fake_ex)
        
        # Gold targets text
        gold_targets_lines = []
        for t in gold.gold_typed_targets:
            gold_targets_lines.append(f"- Action: {t.action_type}, target_belief={t.target_belief_id}, target_cond={t.target_condition_id}, rep_belief={t.replacement_belief_id}, evidence={list(t.evidence_ids)}")
        gold_targets_str = "\n".join(gold_targets_lines) if gold_targets_lines else "None"
        
        # Snapshot statuses
        snapshot_lines = []
        for bid, status in sorted(gold.gold_snapshot.belief_statuses.items()):
            snapshot_lines.append(f"- Belief '{bid}': {status}")
        snapshot_str = "\n".join(snapshot_lines)
        
        # Checklist
        checklist_lines = []
        chk = ep.metadata.get("semantic_checklist", {})
        for name, passed in sorted(chk.items()):
            checklist_lines.append(f"- {name}: {'[PASS]' if passed else '[FAIL]'}")
        checklist_str = "\n".join(checklist_lines)
        
        # Document Markdown Case
        md_lines.append(f"## Case {idx}: {ep.episode_id}")
        md_lines.append(f"- **Domain**: {ep.domain}")
        md_lines.append(f"- **Failure Type**: {ep.failure_type_public_or_controlled}")
        md_lines.append(f"- **Subagent Roles**: {list(ep.subagent_roles)}")
        md_lines.append(f"- **Submissions Count**: {len(ep.submissions)}")
        md_lines.append("")
        md_lines.append("### Method-Visible Prompt Input")
        md_lines.append("```text")
        md_lines.append(user_prompt_text)
        md_lines.append("```")
        md_lines.append("")
        md_lines.append("### Evaluator-Side Intended Actions")
        md_lines.append(gold_targets_str)
        md_lines.append("")
        md_lines.append("### Executable Gold Final Snapshot")
        md_lines.append(snapshot_str)
        md_lines.append("")
        md_lines.append("### Semantic Checklist")
        md_lines.append(checklist_str)
        md_lines.append("")
        md_lines.append("### Review Decision Field")
        md_lines.append("[ ] APPROVE  /  [ ] REVISE  /  [ ] REJECT")
        md_lines.append("")
        md_lines.append("### Reviewer Notes Field")
        md_lines.append("> ")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        # Add to JSONL
        jsonl_row = {
            "episode_id": ep.episode_id,
            "domain": ep.domain,
            "failure_type": ep.failure_type_public_or_controlled,
            "subagent_roles": list(ep.subagent_roles),
            "number_of_submissions": len(ep.submissions),
            "review_status": "pending_human_review",
            "semantic_checklist": chk,
            "gold_snapshot": gold.gold_snapshot.to_dict(),
            "gold_typed_targets": [t.to_dict() for t in gold.gold_typed_targets],
        }
        jsonl_rows.append(jsonl_row)
        
    os.makedirs("outputs", exist_ok=True)
    
    # Write MD review pack
    md_path = "outputs/stagec_smoke7_review_pack.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    # Write JSONL review pack
    jsonl_path = "outputs/stagec_smoke7_review_pack.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in jsonl_rows:
            f.write(json.dumps(r) + "\n")
            
    # Compute checksums
    with open(md_path, "r", encoding="utf-8") as f:
        md_hash = compute_sha256(f.read())
    with open(jsonl_path, "r", encoding="utf-8") as f:
        jsonl_hash = compute_sha256(f.read())
        
    # Write manifest
    manifest = {
        "pack_name": "stagec_smoke7_review_pack",
        "record_count": len(selected_pairs),
        "md_path": md_path,
        "jsonl_path": jsonl_path,
        "md_sha256": md_hash,
        "jsonl_sha256": jsonl_hash,
        "review_status": "pending_human_review",
        "eligible_for_smoke": False,
        "decision_counts": {
            "APPROVE": 0,
            "REVISE": 0,
            "REJECT": 0
        },
        "notes": "Selects 7 pending examples for human review before execution."
    }
    manifest_path = "outputs/stagec_smoke7_review_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    return {
        "record_count": len(selected_pairs),
        "md_path": md_path,
        "jsonl_path": jsonl_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }

def main() -> None:
    res = export_smoke_review_pack()
    print("Stage C Seven-Case Smoke Review Pack exported successfully.")
    print(f"MD Review Pack: {res['md_path']}, Hash: {res['manifest']['md_sha256']}")
    print(f"JSONL Review Pack: {res['jsonl_path']}, Hash: {res['manifest']['jsonl_sha256']}")
    print(f"Manifest: {res['manifest_path']}")
    print(f"Total review records: {res['record_count']}")

if __name__ == "__main__":
    main()

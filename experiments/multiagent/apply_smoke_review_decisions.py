from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List

def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def compute_file_sha256(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return compute_sha256(f.read())

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply human review decisions to Stage C smoke pack (Packet 4F-B)"
    )
    parser.add_argument(
        "--decisions-file",
        type=str,
        default="experiments/multiagent/configs/smoke_review_decisions.json",
        help="Path to the JSON decisions file"
    )
    parser.add_argument(
        "--review-manifest",
        type=str,
        default="outputs/stagec_smoke7_review_manifest.json",
        help="Path to the review pack manifest"
    )
    args = parser.parse_args()

    # 1. Load decisions file
    if not os.path.exists(args.decisions_file):
        print(f"Error: Decisions file not found: {args.decisions_file}")
        sys.exit(1)
    with open(args.decisions_file, "r", encoding="utf-8") as f:
        decisions_data = json.load(f)

    reviewer = decisions_data.get("reviewer")
    reviewed_at = decisions_data.get("reviewed_at")
    source_manifest_sha256 = decisions_data.get("source_manifest_sha256")
    decisions_list = decisions_data.get("decisions", [])

    if not reviewer or not reviewed_at or not source_manifest_sha256:
        print("Error: Decisions JSON must contain 'reviewer', 'reviewed_at', and 'source_manifest_sha256'.")
        sys.exit(1)

    # 2. Load manifest
    if not os.path.exists(args.review_manifest):
        print(f"Error: Manifest file not found: {args.review_manifest}")
        sys.exit(1)
    with open(args.review_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    jsonl_path = manifest.get("jsonl_path")
    md_path = manifest.get("md_path")
    manifest_jsonl_sha256 = manifest.get("jsonl_sha256")

    # Verify matching hash
    current_jsonl_sha256 = compute_file_sha256(jsonl_path)
    # Check manifest file hash as well
    with open(args.review_manifest, "r", encoding="utf-8") as f:
        # Re-encode just to see manifest file content
        manifest_text = f.read()
    manifest_file_sha256 = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()

    # Match against source_manifest_sha256
    hash_matched = (
        source_manifest_sha256 == current_jsonl_sha256 or 
        source_manifest_sha256 == manifest_jsonl_sha256 or
        source_manifest_sha256 == manifest_file_sha256
    )
    if not hash_matched:
        print(f"Error: Decision source_manifest_sha256 '{source_manifest_sha256}' does not match unapproved review pack hash.")
        print(f"Current JSONL SHA-256 is: {current_jsonl_sha256}")
        sys.exit(1)

    # Load unapproved records
    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Verify all episodes have a decision
    episodes_in_pack = {r["episode_id"] for r in records}
    decision_map = {}
    for d in decisions_list:
        ep_id = d.get("episode_id")
        dec = d.get("decision")
        notes = d.get("notes", "")
        if ep_id and dec:
            decision_map[ep_id] = (dec, notes)

    missing_decisions = episodes_in_pack - set(decision_map.keys())
    if missing_decisions:
        print(f"Error: Missing explicit review decisions for: {sorted(list(missing_decisions))}")
        sys.exit(1)

    # Apply decisions to JSONL records
    updated_records = []
    for r in records:
        ep_id = r["episode_id"]
        dec, notes = decision_map[ep_id]
        if dec not in ("APPROVE", "REVISE", "REJECT"):
            print(f"Error: Invalid decision '{dec}' for episode '{ep_id}'. Must be APPROVE, REVISE, or REJECT.")
            sys.exit(1)
            
        r["review_status"] = "approved" if dec == "APPROVE" else dec.lower()
        r["review_provenance"] = {
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "decision": dec,
            "notes": notes,
            "source_manifest_sha256": source_manifest_sha256,
        }
        updated_records.append(r)

    # Apply decisions to outputs/stagec_dev_review_queue_70.jsonl
    queue_path = "outputs/stagec_dev_review_queue_70.jsonl"
    if not os.path.exists(queue_path):
        print(f"Error: Queue file not found: {queue_path}")
        sys.exit(1)
        
    queue_records = []
    with open(queue_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queue_records.append(json.loads(line))
                
    for qr in queue_records:
        ep_id = qr.get("episode_id")
        if ep_id in decision_map:
            dec, notes = decision_map[ep_id]
            qr["review_status"] = "approved" if dec == "APPROVE" else dec.lower()
            qr["review_provenance"] = {
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
                "decision": dec,
                "notes": notes,
                "source_manifest_sha256": source_manifest_sha256,
            }
            
    # Write updated queue JSONL
    with open(queue_path, "w", encoding="utf-8") as f:
        for qr in queue_records:
            f.write(json.dumps(qr) + "\n")
            
    # Write updated pack JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ur in updated_records:
            f.write(json.dumps(ur) + "\n")

    # Update outputs/stagec_smoke7_review_pack.md visualization
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            md_lines = f.read().splitlines()
            
        new_md_lines = []
        current_case_ep = None
        
        for line in md_lines:
            if line.startswith("## Case "):
                # Extract episode_id
                parts = line.split(":")
                if len(parts) >= 2:
                    current_case_ep = parts[1].strip()
            
            if current_case_ep and line.strip() == "### Review Decision Field":
                pass
            elif current_case_ep and "[ ] APPROVE  /  [ ] REVISE  /  [ ] REJECT" in line:
                dec, _ = decision_map[current_case_ep]
                if dec == "APPROVE":
                    line = "[x] APPROVE  /  [ ] REVISE  /  [ ] REJECT"
                elif dec == "REVISE":
                    line = "[ ] APPROVE  /  [x] REVISE  /  [ ] REJECT"
                elif dec == "REJECT":
                    line = "[ ] APPROVE  /  [ ] REVISE  /  [x] REJECT"
            elif current_case_ep and line.startswith("### Reviewer Notes Field"):
                pass
            elif current_case_ep and line.startswith("> "):
                _, notes = decision_map[current_case_ep]
                line = f"> {notes}"
                
            new_md_lines.append(line)
            
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_md_lines) + "\n")

    # Recompute hashes for manifests
    new_jsonl_sha256 = compute_file_sha256(jsonl_path)
    new_md_sha256 = compute_file_sha256(md_path)
    new_queue_sha256 = compute_file_sha256(queue_path)

    # Write stagec_smoke7_review_manifest.json
    manifest["md_sha256"] = new_md_sha256
    manifest["jsonl_sha256"] = new_jsonl_sha256
    manifest["review_status"] = "approved"
    manifest["review_provenance"] = {
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "source_manifest_sha256": source_manifest_sha256,
        "decisions_count": len(decisions_list),
    }
    with open(args.review_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Update stagec_dev_review_manifest.json
    dev_manifest_path = "outputs/stagec_dev_review_manifest.json"
    if os.path.exists(dev_manifest_path):
        with open(dev_manifest_path, "r", encoding="utf-8") as f:
            dev_manifest = json.load(f)
        dev_manifest["jsonl_sha256"] = new_queue_sha256
        # Count approved
        approved_count = sum(1 for qr in queue_records if qr.get("review_status") == "approved")
        dev_manifest["notes"] = f"Expanded 70-example dev review queue. Approved count: {approved_count}/70."
        with open(dev_manifest_path, "w", encoding="utf-8") as f:
            json.dump(dev_manifest, f, indent=2)

    print("[+] Review decisions applied successfully!")
    print(f"    - Reviewer: {reviewer}")
    print(f"    - Reviewed At: {reviewed_at}")
    print(f"    - Updated manifest: {args.review_manifest}")

if __name__ == "__main__":
    main()

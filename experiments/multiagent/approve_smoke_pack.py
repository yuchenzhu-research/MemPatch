from __future__ import annotations

import hashlib
import json
import os

def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def main() -> None:
    smoke7_ids = {
        "ep_expansion_software_engineering_direct_supersession_v1",
        "ep_expansion_research_workflow_stale_propagation_v1",
        "ep_expansion_software_engineering_scope_expansion_v1",
        "ep_expansion_research_workflow_cross_agent_conflict_v1",
        "ep_expansion_software_engineering_temporary_blocker_recovery_v1",
        "ep_expansion_research_workflow_duplicate_evidence_v1",
        "ep_expansion_software_engineering_ambiguous_update_v1",
    }

    # 1. Update outputs/stagec_smoke7_review_pack.md
    md_path = "outputs/stagec_smoke7_review_pack.md"
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        
        # Replace [ ] APPROVE with [x] APPROVE
        md_content = md_content.replace("[ ] APPROVE  /  [ ] REVISE  /  [ ] REJECT", "[x] APPROVE  /  [ ] REVISE  /  [ ] REJECT")
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"[+] Updated {md_path}")
    else:
        print(f"[!] Warning: {md_path} not found.")

    # 2. Update outputs/stagec_smoke7_review_pack.jsonl
    jsonl7_path = "outputs/stagec_smoke7_review_pack.jsonl"
    if os.path.exists(jsonl7_path):
        updated_rows = []
        with open(jsonl7_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("review_status") == "pending_human_review":
                    record["review_status"] = "approved"
                updated_rows.append(record)
        
        with open(jsonl7_path, "w", encoding="utf-8") as f:
            for r in updated_rows:
                f.write(json.dumps(r) + "\n")
        print(f"[+] Updated {jsonl7_path}")
    else:
        print(f"[!] Warning: {jsonl7_path} not found.")

    # 3. Update outputs/stagec_dev_review_queue_70.jsonl
    queue70_path = "outputs/stagec_dev_review_queue_70.jsonl"
    if os.path.exists(queue70_path):
        updated_queue = []
        with open(queue70_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("episode_id") in smoke7_ids:
                    record["review_status"] = "approved"
                updated_queue.append(record)
        
        with open(queue70_path, "w", encoding="utf-8") as f:
            for r in updated_queue:
                f.write(json.dumps(r) + "\n")
        print(f"[+] Updated {queue70_path}")
    else:
        print(f"[!] Warning: {queue70_path} not found.")

    # 4. Recompute hashes and update stagec_smoke7_review_manifest.json
    manifest_smoke7_path = "outputs/stagec_smoke7_review_manifest.json"
    if os.path.exists(manifest_smoke7_path):
        with open(md_path, "r", encoding="utf-8") as f:
            md_hash = compute_sha256(f.read())
        with open(jsonl7_path, "r", encoding="utf-8") as f:
            jsonl_hash = compute_sha256(f.read())
            
        with open(manifest_smoke7_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            
        manifest["md_sha256"] = md_hash
        manifest["jsonl_sha256"] = jsonl_hash
        manifest["review_status"] = "approved"
        manifest["notes"] = "Approved Stage C smoke examples ready for prompt execution."
        
        with open(manifest_smoke7_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"[+] Updated {manifest_smoke7_path}")

    # 5. Recompute hash and update stagec_dev_review_manifest.json
    manifest_queue70_path = "outputs/stagec_dev_review_manifest.json"
    if os.path.exists(manifest_queue70_path):
        with open(queue70_path, "r", encoding="utf-8") as f:
            queue_hash = compute_sha256(f.read())
            
        with open(manifest_queue70_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            
        manifest["jsonl_sha256"] = queue_hash
        
        # Check how many are approved in the queue
        approved_count = 0
        with open(queue70_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    if json.loads(line).get("review_status") == "approved":
                        approved_count += 1
        
        manifest["notes"] = f"Expanded 70-example dev review queue. Approved count: {approved_count}/70."
        
        with open(manifest_queue70_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"[+] Updated {manifest_queue70_path}")

if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import pytest
import tempfile
from unittest.mock import patch
from experiments.multiagent.select_prompt_smoke_examples import select_smoke_examples
from experiments.multiagent.apply_smoke_review_decisions import main as apply_main

def test_select_smoke_examples_provenance_enforced(tmp_path):
    # Setup eligible manifest
    manifest_path = tmp_path / "stagec_smoke7_review_manifest.json"
    manifest_path.write_text(json.dumps({
        "eligible_for_smoke": True,
        "review_status": "approved_for_smoke"
    }))
    
    review_file = tmp_path / "review.jsonl"
    review_file.write_text(json.dumps({
        "episode_id": "ep_test_1",
        "failure_type": "direct_supersession",
        "review_status": "approved",
    }) + "\n")
    
    with patch("experiments.multiagent.select_prompt_smoke_examples.MANIFEST_PATH", str(manifest_path)):
        with pytest.raises(SystemExit) as excinfo:
            select_smoke_examples(str(review_file), confirm_live_run=True)
        assert excinfo.value.code == 1

def test_select_smoke_examples_provenance_passes_when_valid(tmp_path):
    manifest_path = tmp_path / "stagec_smoke7_review_manifest.json"
    manifest_path.write_text(json.dumps({
        "eligible_for_smoke": True,
        "review_status": "approved_for_smoke"
    }))
    
    review_file = tmp_path / "review.jsonl"
    review_file.write_text(json.dumps({
        "episode_id": "ep_test_1",
        "failure_type": "direct_supersession",
        "review_status": "approved",
        "review_provenance": {
            "reviewer": "Test Reviewer",
            "reviewed_at": "2026-05-30T00:00:00Z",
            "source_manifest_sha256": "dummy_sha"
        }
    }) + "\n")
    
    with patch("experiments.multiagent.select_prompt_smoke_examples.MANIFEST_PATH", str(manifest_path)):
        selected = select_smoke_examples(str(review_file), confirm_live_run=True)
        assert len(selected) == 1
        assert selected[0]["episode_id"] == "ep_test_1"

def test_select_smoke_examples_fails_when_manifest_not_eligible(tmp_path):
    manifest_path = tmp_path / "stagec_smoke7_review_manifest.json"
    manifest_path.write_text(json.dumps({
        "eligible_for_smoke": False,
        "review_status": "requires_revision"
    }))
    
    review_file = tmp_path / "review.jsonl"
    review_file.write_text(json.dumps({
        "episode_id": "ep_test_1",
        "failure_type": "direct_supersession",
        "review_status": "approved",
        "review_provenance": {
            "reviewer": "Test Reviewer",
            "reviewed_at": "2026-05-30T00:00:00Z",
            "source_manifest_sha256": "dummy_sha"
        }
    }) + "\n")
    
    with patch("experiments.multiagent.select_prompt_smoke_examples.MANIFEST_PATH", str(manifest_path)):
        with pytest.raises(SystemExit) as excinfo:
            select_smoke_examples(str(review_file), confirm_live_run=True)
        assert excinfo.value.code == 1

def test_apply_decisions_states(tmp_path):
    # Setup test review pack files
    md_file = tmp_path / "pack.md"
    md_file.write_text("## Case 1: ep1\n[ ] APPROVE  /  [ ] REVISE  /  [ ] REJECT\n> \n## Case 2: ep2\n[ ] APPROVE  /  [ ] REVISE  /  [ ] REJECT\n> ")
    
    jsonl_file = tmp_path / "pack.jsonl"
    records = [
        {"episode_id": "ep1", "review_status": "pending_human_review", "failure_type": "direct_supersession"},
        {"episode_id": "ep2", "review_status": "pending_human_review", "failure_type": "stale_propagation"}
    ]
    with open(jsonl_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    import hashlib
    jsonl_content = jsonl_file.read_text()
    jsonl_hash = hashlib.sha256(jsonl_content.encode("utf-8")).hexdigest()
    
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({
        "jsonl_path": str(jsonl_file),
        "md_path": str(md_file),
        "jsonl_sha256": jsonl_hash
    }))
    
    # 70 cases mock
    queue_file = tmp_path / "queue.jsonl"
    with open(queue_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    dev_manifest_file = tmp_path / "dev_manifest.json"
    dev_manifest_file.write_text(json.dumps({
        "jsonl_sha256": "initial_sha",
        "notes": ""
    }))
    
    # Mock files inside the script path
    # Test 1: All Approved
    decisions_1 = tmp_path / "decisions_1.json"
    decisions_1.write_text(json.dumps({
        "reviewer": "Yuchen",
        "reviewed_at": "2026",
        "source_manifest_sha256": jsonl_hash,
        "decisions": [
            {"episode_id": "ep1", "decision": "APPROVE", "notes": "ok1"},
            {"episode_id": "ep2", "decision": "APPROVE", "notes": "ok2"}
        ]
    }))
    
    with patch("sys.argv", ["", "--decisions-file", str(decisions_1), "--review-manifest", str(manifest_file)]):
        with patch("experiments.multiagent.apply_smoke_review_decisions.QUEUE_PATH", str(queue_file)):
            with patch("experiments.multiagent.apply_smoke_review_decisions.DEV_MANIFEST_PATH", str(dev_manifest_file)):
                apply_main()
            
    # Read back manifest
    with open(manifest_file, "r") as f:
        m = json.load(f)
    assert m["review_status"] == "approved_for_smoke"
    assert m["eligible_for_smoke"] is True
    assert m["decision_counts"] == {"APPROVE": 2, "REVISE": 0, "REJECT": 0}

    # Test 2: One Revise
    decisions_2 = tmp_path / "decisions_2.json"
    # We need to re-read and compute hash because pack.jsonl is modified
    jsonl_hash_2 = hashlib.sha256(jsonl_file.read_text().encode("utf-8")).hexdigest()
    decisions_2.write_text(json.dumps({
        "reviewer": "Yuchen",
        "reviewed_at": "2026",
        "source_manifest_sha256": jsonl_hash_2,
        "decisions": [
            {"episode_id": "ep1", "decision": "APPROVE", "notes": "ok1"},
            {"episode_id": "ep2", "decision": "REVISE", "notes": "need revision"}
        ]
    }))
    
    with patch("sys.argv", ["", "--decisions-file", str(decisions_2), "--review-manifest", str(manifest_file)]):
        with patch("experiments.multiagent.apply_smoke_review_decisions.QUEUE_PATH", str(queue_file)):
            with patch("experiments.multiagent.apply_smoke_review_decisions.DEV_MANIFEST_PATH", str(dev_manifest_file)):
                apply_main()
            
    with open(manifest_file, "r") as f:
        m = json.load(f)
    assert m["review_status"] == "requires_revision"
    assert m["eligible_for_smoke"] is False
    assert m["decision_counts"] == {"APPROVE": 1, "REVISE": 1, "REJECT": 0}

    # Test 3: One Reject
    decisions_3 = tmp_path / "decisions_3.json"
    jsonl_hash_3 = hashlib.sha256(jsonl_file.read_text().encode("utf-8")).hexdigest()
    decisions_3.write_text(json.dumps({
        "reviewer": "Yuchen",
        "reviewed_at": "2026",
        "source_manifest_sha256": jsonl_hash_3,
        "decisions": [
            {"episode_id": "ep1", "decision": "REJECT", "notes": "bad"},
            {"episode_id": "ep2", "decision": "APPROVE", "notes": "ok"}
        ]
    }))
    
    with patch("sys.argv", ["", "--decisions-file", str(decisions_3), "--review-manifest", str(manifest_file)]):
        with patch("experiments.multiagent.apply_smoke_review_decisions.QUEUE_PATH", str(queue_file)):
            with patch("experiments.multiagent.apply_smoke_review_decisions.DEV_MANIFEST_PATH", str(dev_manifest_file)):
                apply_main()
            
    with open(manifest_file, "r") as f:
        m = json.load(f)
    assert m["review_status"] == "not_approved"
    assert m["eligible_for_smoke"] is False
    assert m["decision_counts"] == {"APPROVE": 1, "REVISE": 0, "REJECT": 1}

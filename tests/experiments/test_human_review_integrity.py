from __future__ import annotations

import json
import os
import pytest
import tempfile
import sys
from unittest.mock import patch
from experiments.multiagent.select_prompt_smoke_examples import select_smoke_examples

def test_select_smoke_examples_provenance_enforced():
    # Test that select_smoke_examples rejects items with "review_status" = "approved" but lacking "review_provenance"
    dummy_data = [
        {
            "episode_id": "ep_test_1",
            "failure_type": "direct_supersession",
            "review_status": "approved",
            # No review_provenance
        }
    ]
    
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in dummy_data:
            f.write(json.dumps(r) + "\n")
        temp_path = f.name
        
    try:
        with pytest.raises(SystemExit) as excinfo:
            select_smoke_examples(temp_path, confirm_live_run=True)
        # Should exit with code 1
        assert excinfo.value.code == 1
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_select_smoke_examples_provenance_passes_when_valid():
    dummy_data = [
        {
            "episode_id": "ep_test_1",
            "failure_type": "direct_supersession",
            "review_status": "approved",
            "review_provenance": {
                "reviewer": "Test Reviewer",
                "reviewed_at": "2026-05-30T00:00:00Z",
                "source_manifest_sha256": "dummy_sha"
            }
        }
    ]
    
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in dummy_data:
            f.write(json.dumps(r) + "\n")
        temp_path = f.name
        
    try:
        selected = select_smoke_examples(temp_path, confirm_live_run=True)
        assert len(selected) == 1
        assert selected[0]["episode_id"] == "ep_test_1"
        assert selected[0]["review_provenance"]["reviewer"] == "Test Reviewer"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

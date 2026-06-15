"""Test suite for MemPatch Consensus Mode."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from mempatch.revision.runtime.dpa_runtime import run_consensus
from mempatch.revision.runtime.scenario_revision import build_scenario_revision_view

def test_consensus_mode_success():
    test_scenarios_path = Path("local/data/mempatch/test/scenarios.jsonl")
    with test_scenarios_path.open("r", encoding="utf-8") as f:
        scenario = json.loads(f.readline())
        
    view = build_scenario_revision_view(scenario)
    
    # Matching proposals
    proposal_text_1 = json.dumps([
        {
            "action_type": "NO_REVISION",
            "target_belief_id": None,
            "target_condition_id": None,
            "replacement_belief_id": None,
            "evidence_ids": [view.new_evidence.evidence_id],
            "rationale": "noop test",
        }
    ])
    
    proposal_text_2 = json.dumps([
        {
            "action_type": "NO_REVISION",
            "target_belief_id": None,
            "target_condition_id": None,
            "replacement_belief_id": None,
            "evidence_ids": [view.new_evidence.evidence_id],
            "rationale": "noop test", # identical actions
        }
    ])
    
    res = run_consensus(view, [proposal_text_1, proposal_text_2])
    assert res.parse_result.schema_valid
    assert len(res.admitted_actions) == 0 # noop actions
    assert len(res.engine_errors) == 0

def test_consensus_mode_disagreement():
    test_scenarios_path = Path("local/data/mempatch/test/scenarios.jsonl")
    with test_scenarios_path.open("r", encoding="utf-8") as f:
        scenario = json.loads(f.readline())
        
    view = build_scenario_revision_view(scenario)
    
    # Conflicting proposals
    proposal_text_1 = json.dumps([
        {
            "action_type": "NO_REVISION",
            "target_belief_id": None,
            "target_condition_id": None,
            "replacement_belief_id": None,
            "evidence_ids": [view.new_evidence.evidence_id],
            "rationale": "noop test",
        }
    ])
    
    # Supercedes action
    proposal_text_2 = json.dumps([
        {
            "action_type": "SUPERSEDES",
            "target_belief_id": "m-case-3501-target",
            "target_condition_id": None,
            "replacement_belief_id": "m-case-3501-target_renamed_val",
            "evidence_ids": [view.new_evidence.evidence_id],
            "rationale": "malformed test action",
        }
    ])
    
    res = run_consensus(view, [proposal_text_1, proposal_text_2])
    assert not res.parse_result.schema_valid
    assert len(res.engine_errors) == 1
    assert res.engine_errors[0].code == "CONSENSUS_DISAGREEMENT"

"""Test suite for Reference Transition Semantics."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from mempatch.reference_semantics.normal_form import evaluate_normal_form

def test_reference_transition_semantics():
    local_path = Path("local/data/mempatch/test/scenarios.jsonl")
    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            scenarios = [json.loads(line) for line in f if line.strip()]
    else:
        fixture_path = Path(__file__).parent / "fixtures" / "smoke_scenarios.jsonl"
        with fixture_path.open("r", encoding="utf-8") as f:
            scenarios = [json.loads(line) for line in f if line.strip()]

    # Test a subset of scenarios for fast verification
    import random
    rng = random.Random(42)
    sample_size = min(len(scenarios), 50)
    sample_scenarios = rng.sample(scenarios, sample_size)

    for idx, scenario in enumerate(sample_scenarios):
        sid = scenario.get("scenario_id", "")
        gold = scenario.get("hidden_gold", {})
        
        # Run reference semantics evaluator
        nf = evaluate_normal_form(scenario)
        
        # Verify decision
        assert nf["decision"] == gold.get("expected_decision"), f"Decision mismatch at {sid}: {nf['decision']} vs {gold.get('expected_decision')}"
        
        # Verify memory states
        expected_state = gold.get("expected_memory_state", {})
        for mid, status in expected_state.items():
            assert nf["final_state"].get(mid) == status, f"Memory state mismatch at {sid} for {mid}: {nf['final_state'].get(mid)} vs {status}"
            
        # Verify answer key facts
        assert nf["answer_key_facts"] == gold.get("expected_answer"), f"Answer key facts mismatch at {sid}"
        
        # Verify derivation trace is populated for non-use-current decisions
        if nf["decision"] != "use_current_memory":
            assert len(nf["derivation_trace"]) > 0, f"Expected non-empty derivation trace for decision {nf['decision']} at {sid}"
            # Verify trace structure
            for step in nf["derivation_trace"]:
                assert "rule_id" in step
                assert "target_record_id" in step
                assert "evidence_ids" in step
                assert "pre_state" in step
                assert "post_state" in step
                assert "priority" in step
                assert "reason_code" in step

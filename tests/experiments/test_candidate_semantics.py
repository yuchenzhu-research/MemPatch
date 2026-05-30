from __future__ import annotations

import copy
import pytest
from experiments.multiagent.dev_expansion import generate_expanded_episodes
from experiments.multiagent.legacy.validate_candidate_semantics import validate_episode

def test_all_70_candidates_pass_validation():
    episodes_with_gold = generate_expanded_episodes()
    assert len(episodes_with_gold) == 70
    
    for ep, gold in episodes_with_gold:
        is_pass, detail = validate_episode(ep, gold)
        assert is_pass, f"Episode {ep.episode_id} failed executable consistency verification: {detail['mismatches']}"

def test_deliberate_inconsistency_fails_validation():
    episodes_with_gold = generate_expanded_episodes()
    ep, gold = episodes_with_gold[0]
    
    # Deliberately mutate gold expectations to cause inconsistency
    inconsistent_gold = copy.deepcopy(gold)
    belief_id = list(inconsistent_gold.gold_snapshot.belief_statuses.keys())[0]
    original_status = inconsistent_gold.gold_snapshot.belief_statuses[belief_id]
    
    # Mutate to opposite status
    wrong_status = "BLOCKED" if original_status == "AUTHORIZED" or original_status == "SUPERSEDED" else "AUTHORIZED"
    inconsistent_gold.gold_snapshot.belief_statuses[belief_id] = wrong_status
    
    is_pass, detail = validate_episode(ep, inconsistent_gold)
    assert not is_pass
    assert belief_id in detail["mismatches"]
    assert detail["mismatches"][belief_id]["expected"] == wrong_status
    assert detail["mismatches"][belief_id]["actual"] == original_status

def test_downstream_task_expectations_are_case_local():
    episodes_with_gold = generate_expanded_episodes()
    for ep, gold in episodes_with_gold:
        for task in ep.downstream_tasks:
            # Downstream expected answer or action must match one of the active statuses or be BLOCKED/UNRESOLVED
            expected = task.expected_answer_or_action
            assert expected in ("AUTHORIZED", "BLOCKED", "UNRESOLVED")
            
            # Ensure no variable leaks (e.g. check domain appropriate variables)
            # The task_id must belong to this episode
            assert task.task_id == f"task_{ep.episode_id}"

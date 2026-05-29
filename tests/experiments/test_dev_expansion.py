from __future__ import annotations

import pytest
from experiments.multiagent.dev_expansion import generate_expanded_episodes, DOMAINS, FAILURE_TYPES

def test_dev_expansion_generation() -> None:
    episodes_with_gold = generate_expanded_episodes()
    
    assert len(episodes_with_gold) == 70
    
    seen_combinations = set()
    
    for ep, gold in episodes_with_gold:
        # Check basic fields
        assert ep.split == "development_candidate"
        assert ep.domain in DOMAINS
        assert ep.failure_type_public_or_controlled in FAILURE_TYPES
        
        # Check metadata review status
        meta = ep.metadata
        assert meta.get("review_status") == "pending_human_review"
        assert meta.get("training_eligible") is False
        assert meta.get("scientific_status") == "not_evaluated"
        assert meta.get("label_source") == "template_authored_pending_review"
        
        # Ensure gold metadata matches episode metadata
        assert gold.metadata == meta
        assert gold.split == "development_candidate" if hasattr(gold, "split") else True
        assert gold.failure_type == ep.failure_type_public_or_controlled
        
        # Track combinations to verify exact grid coverage (2 domains x 7 failures x 5 variants)
        comb_key = (ep.domain, ep.failure_type_public_or_controlled)
        seen_combinations.add(comb_key)
        
        # Verify target bounds
        for t in gold.gold_typed_targets:
            assert t.submission_id in [s.submission_id for s in ep.submissions]
            if t.action_type == "NO_REVISION":
                assert t.target_belief_id is None
                assert t.target_condition_id is None
                assert t.replacement_belief_id is None
            elif t.action_type == "SUPERSEDES":
                assert t.target_belief_id is not None
                assert t.replacement_belief_id is not None
            elif t.action_type in ("BLOCKS", "RELEASES"):
                assert t.target_condition_id is not None
            else: # UNCERTAIN, REAFFIRMS
                assert t.target_belief_id is not None
                
    assert len(seen_combinations) == len(DOMAINS) * len(FAILURE_TYPES)


def test_select_smoke_examples_error(tmp_path) -> None:
    # 1. Test when no approved examples exist
    review_file = tmp_path / "review.jsonl"
    review_file.write_text(
        '{"episode_id": "ep_1", "review_status": "pending_human_review", "failure_type": "direct_supersession"}\n'
    )
    
    from experiments.multiagent.select_prompt_smoke_examples import select_smoke_examples
    with pytest.raises(SystemExit):
        select_smoke_examples(str(review_file), confirm_live_run=True)
        
    # 2. Test when approved exists but not confirmed
    review_file.write_text(
        '{"episode_id": "ep_1", "review_status": "approved", "failure_type": "direct_supersession"}\n'
    )
    with pytest.raises(SystemExit):
        select_smoke_examples(str(review_file), confirm_live_run=False)
        
    # 3. Test successful selection
    selected = select_smoke_examples(str(review_file), confirm_live_run=True)
    assert len(selected) == 1
    assert selected[0]["episode_id"] == "ep_1"

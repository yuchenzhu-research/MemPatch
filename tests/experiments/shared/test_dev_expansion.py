from __future__ import annotations

import pytest
from retracemem.evaluation.multiagent.data.dev_expansion import generate_expanded_episodes, DOMAINS, FAILURE_TYPES

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
                
        # Verify Checklist
        checklist = meta.get("semantic_checklist")
        assert checklist is not None
        assert checklist["has_visible_new_evidence"] is True
        assert checklist["typed_target_ids_visible"] is True
        assert checklist["downstream_task_defined"] is True
        assert meta.get("semantic_validation_status", {}).get("passes_structural_checks") is True

        if ep.failure_type_public_or_controlled == "temporary_blocker_recovery":
            assert len(ep.submissions) == 3
            assert gold.gold_typed_targets[0].action_type == "BLOCKS"
            assert gold.gold_typed_targets[1].action_type == "RELEASES"
            
        elif ep.failure_type_public_or_controlled == "scope_expansion":
            has_prot = any(b.belief_id.endswith("protected") for b in ep.submissions[-1].candidate_beliefs)
            assert has_prot is True
            assert gold.gold_snapshot.belief_statuses[f"b_{ep.episode_id}_protected"] == "AUTHORIZED"
            
        elif ep.failure_type_public_or_controlled == "duplicate_evidence":
            assert "duplicates_evidence_id" in ep.submissions[-1].metadata
            assert gold.gold_typed_targets[0].action_type == "NO_REVISION"
            
        elif ep.failure_type_public_or_controlled == "ambiguous_update":
            assert gold.gold_typed_targets[0].action_type == "UNCERTAIN"
            assert any("uncertainty_cue" in ev.metadata for ev in ep.submissions[-1].evidence_context)

    assert len(seen_combinations) == len(DOMAINS) * len(FAILURE_TYPES)


def test_select_smoke_examples_error(tmp_path) -> None:
    import json
    from unittest.mock import patch
    # 1. Test when no approved examples exist
    review_file = tmp_path / "review.jsonl"
    review_file.write_text(
        '{"episode_id": "ep_1", "review_status": "pending_human_review", "failure_type": "direct_supersession"}\n'
    )
    
    # Create mock eligible manifest
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "eligible_for_smoke": True,
        "review_status": "approved_for_smoke"
    }))
    
    from experiments.multiagent.select_prompt_smoke_examples import select_smoke_examples
    with patch("experiments.multiagent.select_prompt_smoke_examples.MANIFEST_PATH", str(manifest_path)):
        with pytest.raises(SystemExit):
            select_smoke_examples(str(review_file), confirm_live_run=True)
        
    # 2. Test when approved exists but not confirmed
    review_file.write_text(
        '{"episode_id": "ep_1", "review_status": "approved", "failure_type": "direct_supersession", "review_provenance": {"reviewer": "Yuchen Zhu", "reviewed_at": "2026-05-30T00:00:00Z", "source_manifest_sha256": "dummy_sha"}}\n'
    )
    with patch("experiments.multiagent.select_prompt_smoke_examples.MANIFEST_PATH", str(manifest_path)):
        with pytest.raises(SystemExit):
            select_smoke_examples(str(review_file), confirm_live_run=False)
        
    # 3. Test successful selection
    with patch("experiments.multiagent.select_prompt_smoke_examples.MANIFEST_PATH", str(manifest_path)):
        selected = select_smoke_examples(str(review_file), confirm_live_run=True)
    assert len(selected) == 1
    assert selected[0]["episode_id"] == "ep_1"


def test_smoke_runner_preflight_and_live(tmp_path) -> None:
    import json
    config_file = tmp_path / "smoke_config.json"
    config_file.write_text(json.dumps({
        "run_config": {"run_id_prefix": "test_smoke", "requires_explicit_user_approval": True},
        "model_config": {"provider": "<openai>", "backbone_model": "<select_before_run>"},
        "dataset_config": {"split": "development_only"}
    }))
    
    review_file = tmp_path / "review.jsonl"
    review_file.write_text(
        '{"episode_id": "ep_1", "review_status": "pending_human_review", "failure_type": "direct_supersession"}\n'
    )
    
    from experiments.multiagent.legacy.run_stagec_prompt_smoke import run_preflight, run_live
    
    run_preflight(str(config_file), str(review_file))
    
    with pytest.raises(SystemExit):
        run_live(str(config_file), str(review_file), confirm_live_run=True)
        
    with pytest.raises(SystemExit):
        run_live(str(config_file), str(review_file), confirm_live_run=False)

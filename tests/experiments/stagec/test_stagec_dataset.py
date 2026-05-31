from __future__ import annotations

import pytest
from experiments.multiagent.stagec_dataset import build_stagec_dataset, validate_training_example
from experiments.multiagent.contracts import StageCTrainingExample, TypedRevisionTarget

def test_build_stagec_dataset_success():
    examples = build_stagec_dataset()
    
    # 14 episodes, some have 2 submissions, so we expect around 20-25 examples
    assert len(examples) > 14
    
    # Verify properties of all examples
    for ex in examples:
        assert ex.split == "development_only"
        assert ex.label_source == "human_authored_seed"
        assert ex.metadata["scientific_status"] == "pipeline_validation_only"
        assert ex.metadata["training_eligible"] is False
        assert ex.metadata["contains_gold_in_user_input"] is False
        
        # Verify method_visible_input has no candidate_edges or gold
        assert not hasattr(ex.method_visible_input, "candidate_edges")
        
        # Verify canonical actions
        for t in ex.targets:
            assert t.action_type in {"SUPERSEDES", "BLOCKS", "RELEASES", "UNCERTAIN", "REAFFIRMS", "NO_REVISION"}
            if t.action_type == "SUPERSEDES":
                assert t.target_belief_id is not None
                assert t.replacement_belief_id is not None


def test_validate_training_example_failure():
    # Construct invalid example to trigger validations
    from experiments.multiagent.episodes_fc_dev import get_fc_dev_episodes
    ep, gold, _ = get_fc_dev_episodes()[0]
    sub = ep.submissions[0]
    
    # 1. Invalid split
    ex_invalid_split = StageCTrainingExample(
        example_id="ex_1", episode_id=ep.episode_id, submission_id=sub.submission_id,
        method_visible_input=sub, targets=(), split="train",
        domain=ep.domain, failure_type=gold.failure_type, label_source="test",
        metadata={"scientific_status": "pipeline_validation_only"}
    )
    with pytest.raises(ValueError, match="must have split='development_only'"):
        validate_training_example(ex_invalid_split)
        
    # 2. Invalid scientific status
    ex_invalid_status = StageCTrainingExample(
        example_id="ex_2", episode_id=ep.episode_id, submission_id=sub.submission_id,
        method_visible_input=sub, targets=(), split="development_only",
        domain=ep.domain, failure_type=gold.failure_type, label_source="test",
        metadata={"scientific_status": "official_ready"}
    )
    with pytest.raises(ValueError, match="must be tagged with scientific_status='pipeline_validation_only'"):
        validate_training_example(ex_invalid_status)

    # 3. Invalid target action
    bad_target = TypedRevisionTarget(
        submission_id=sub.submission_id,
        action_type="OVERWRITE_ALL_BELIEFS", # non-canonical
        target_belief_id="b1"
    )
    ex_bad_action = StageCTrainingExample(
        example_id="ex_3", episode_id=ep.episode_id, submission_id=sub.submission_id,
        method_visible_input=sub, targets=(bad_target,), split="development_only",
        domain=ep.domain, failure_type=gold.failure_type, label_source="test",
        metadata={"scientific_status": "pipeline_validation_only"}
    )
    with pytest.raises(ValueError, match="is not canonical"):
        validate_training_example(ex_bad_action)

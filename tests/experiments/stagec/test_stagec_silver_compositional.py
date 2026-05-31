from __future__ import annotations

import pytest
from experiments.multiagent.export_stagec_silver_compositional import (
    generate_new_silver_v1_episodes,
    TRAIN_AUGMENTATIONS,
    apply_replacements,
)


def test_generate_new_silver_v1_episodes():
    episodes = generate_new_silver_v1_episodes()
    # 2 domains * 7 patterns * 5 variants = 70 base episodes
    assert len(episodes) == 70

    # Verify all expected patterns are present
    patterns = {ep.failure_type_public_or_controlled for ep, _ in episodes}
    assert "reaffirms_only" in patterns
    assert "grounding_hard" in patterns
    assert "multi_action_supersedes_blocks" in patterns
    assert "multi_action_supersedes_releases" in patterns
    assert "multi_action_blocks_uncertain" in patterns
    assert "evidence_conflict" in patterns
    assert "target_ambiguity" in patterns

    # Verify no STALE/CUPMem in generated IDs
    for ep, gold in episodes:
        assert "stale" not in ep.episode_id.lower() or "stale_propagation" in ep.episode_id.lower()
        assert "cupmem" not in ep.episode_id.lower()


def test_apply_replacements():
    text = "Deploying to Staging and Production servers."
    mapping = {
        "Staging": "QA",
        "Production": "Live",
    }
    replaced = apply_replacements(text, mapping)
    assert replaced == "Deploying to QA and Live servers."


def test_train_augmentations():
    assert len(TRAIN_AUGMENTATIONS) == 8
    # Assert base augmentation is empty replacement mapping
    assert TRAIN_AUGMENTATIONS[0][0] == "base"
    assert TRAIN_AUGMENTATIONS[0][1] == {}

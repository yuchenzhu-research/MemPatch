"""Archived: Stage C prompt-smoke runner preflight/live gating.

Relocated from tests/experiments/shared/test_dev_expansion.py because it
exercises the archived runner experiments.archive.legacy.run_stagec_prompt_smoke,
which is not part of the canonical Stage A/B/C evaluation path.
"""
from __future__ import annotations

import json

import pytest


def test_smoke_runner_preflight_and_live(tmp_path) -> None:
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

    from experiments.archive.legacy.run_stagec_prompt_smoke import run_preflight, run_live

    run_preflight(str(config_file), str(review_file))

    with pytest.raises(SystemExit):
        run_live(str(config_file), str(review_file), confirm_live_run=True)

    with pytest.raises(SystemExit):
        run_live(str(config_file), str(review_file), confirm_live_run=False)

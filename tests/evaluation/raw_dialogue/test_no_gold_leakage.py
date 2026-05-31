"""No-gold-leakage audit for the raw-dialogue pipeline.

These tests are self-contained: they regenerate a small synthetic dataset and
all derived training artifacts into a temporary directory, then assert that no
gold revision actions or evaluator-only final statuses leak into any
method-visible input. They must actually run on a clean checkout (no skips), so
the leakage guarantees are verified on every test invocation.
"""
import json
import os
import subprocess
import sys

import pytest

from retracemem.evaluation.raw_dialogue.contracts import DialogueExtractionTarget

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _run_script(rel_path, *args):
    res = subprocess.run(
        [sys.executable, os.path.join("scripts", rel_path), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert res.returncode == 0, f"{rel_path} failed: {res.stderr}"


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    """Build synth dataset + SFT exports + RL rollouts once for the module."""
    d = tmp_path_factory.mktemp("leakage")
    synth = d / "raw_dialogue_synth.jsonl"
    graph_sft = d / "graph_extractor_sft.jsonl"
    revision_sft = d / "typed_revision_sft.jsonl"
    rollouts = d / "dpa_rl_rollouts.jsonl"

    _run_script("build_raw_dialogue_synth.py", "--out", str(synth), "--n", "26", "--seed", "7")
    _run_script("export_graph_extractor_sft.py", "--in-file", str(synth), "--out-file", str(graph_sft))
    _run_script("export_revision_proposer_sft.py", "--in-file", str(synth), "--out-file", str(revision_sft))
    _run_script("build_dpa_reward_rollouts.py", "--in-file", str(synth), "--out-file", str(rollouts))

    return {
        "synth": synth,
        "graph_sft": graph_sft,
        "revision_sft": revision_sft,
        "rollouts": rollouts,
    }


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_no_gold_leakage_in_method_visible_raw_dialogue(artifacts):
    # The method-visible dialogue block itself must not contain any reference to
    # gold final statuses or gold action labels.
    rows = _read_jsonl(artifacts["synth"])
    assert rows, "synth dataset is empty"
    for data in rows:
        target = DialogueExtractionTarget.from_dict(data)
        for u in target.dialogue.utterances:
            text = u.text.upper()
            assert "AUTHORIZED" not in text
            assert "SUPERSEDED" not in text
            assert "BLOCKED" not in text
            assert "UNRESOLVED" not in text
            assert "GOLD_ACTIONS" not in text


def test_graph_extractor_sft_target_separated_from_input(artifacts):
    rows = _read_jsonl(artifacts["graph_sft"])
    assert rows, "graph SFT dataset is empty"
    for row in rows:
        messages = row["messages"]
        user_content = messages[1]["content"]
        assistant_content = messages[2]["content"]

        # The input (user) must NOT contain the target (gold_graph) representation
        assert "gold_graph" not in user_content
        # The assistant must hold the graph target
        graph_data = json.loads(assistant_content)
        assert "evidence_nodes" in graph_data
        assert "belief_nodes" in graph_data


def test_revision_proposer_sft_target_separated_from_input(artifacts):
    rows = _read_jsonl(artifacts["revision_sft"])
    assert rows, "revision SFT dataset is empty"
    for row in rows:
        messages = row["messages"]
        user_content = messages[1]["content"]
        assistant_content = messages[2]["content"]

        # The input (user) must NOT contain the gold targets
        assert "gold_actions" not in user_content
        assert "gold_final_statuses" not in user_content

        # Assistant content holds the gold actions list
        actions = json.loads(assistant_content)
        assert isinstance(actions, list)


def test_reward_rollout_uses_gold_only_in_training_context(artifacts):
    rows = _read_jsonl(artifacts["rollouts"])
    assert rows, "rollout dataset is empty"
    for row in rows:
        # The model prompt input does not contain the evaluation labels
        assert "gold_final_statuses" not in row["prompt_input"]
        assert "reward_breakdown" not in row["prompt_input"]

        # The gold labels reside strictly in evaluator-only properties
        assert "gold_final_statuses" in row
        assert "reward_breakdown" in row

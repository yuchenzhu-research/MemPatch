import json
import os
import pytest

from retracemem.evaluation.raw_dialogue.contracts import DialogueExtractionTarget


def test_no_gold_leakage_in_method_visible_raw_dialogue():
    # Load synthetic dataset and check that the method-visible dialogue block itself
    # does not contain any reference to gold actions or final statuses.
    in_file = "outputs/raw_dialogue_synth.jsonl"
    if not os.path.exists(in_file):
        pytest.skip("outputs/raw_dialogue_synth.jsonl not found. Run synth generator first.")

    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            target = DialogueExtractionTarget.from_dict(data)
            
            # The method visible dialogue utterances
            for u in target.dialogue.utterances:
                text = u.text.upper()
                assert "AUTHORIZED" not in text
                assert "SUPERSEDED" not in text
                assert "BLOCKED" not in text
                assert "UNRESOLVED" not in text
                assert "GOLD_ACTIONS" not in text


def test_graph_extractor_sft_target_separated_from_input():
    sft_file = "outputs/graph_extractor_sft.jsonl"
    if not os.path.exists(sft_file):
        pytest.skip("outputs/graph_extractor_sft.jsonl not found.")

    with open(sft_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row["messages"]
            
            user_content = messages[1]["content"]
            assistant_content = messages[2]["content"]
            
            # The input (user) must NOT contain the target (gold_graph) representation
            assert "gold_graph" not in user_content
            # The assistant must have the graph target
            graph_data = json.loads(assistant_content)
            assert "evidence_nodes" in graph_data
            assert "belief_nodes" in graph_data


def test_revision_proposer_sft_target_separated_from_input():
    sft_file = "outputs/typed_revision_sft.jsonl"
    if not os.path.exists(sft_file):
        pytest.skip("outputs/typed_revision_sft.jsonl not found.")

    with open(sft_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row["messages"]
            
            user_content = messages[1]["content"]
            assistant_content = messages[2]["content"]
            
            # The input (user) must NOT contain the target gold_actions
            assert "gold_actions" not in user_content
            assert "gold_final_statuses" not in user_content
            
            # Assistant content contains the gold actions list
            actions = json.loads(assistant_content)
            assert isinstance(actions, list)


def test_reward_rollout_uses_gold_only_in_training_context():
    rollout_file = "outputs/dpa_rl_rollouts.jsonl"
    if not os.path.exists(rollout_file):
        pytest.skip("outputs/dpa_rl_rollouts.jsonl not found.")

    with open(rollout_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            
            # The model prompt input does not contain the evaluation labels
            assert "gold_final_statuses" not in row["prompt_input"]
            assert "reward_breakdown" not in row["prompt_input"]
            
            # The gold labels reside strictly in evaluator properties
            assert "gold_final_statuses" in row
            assert "reward_breakdown" in row

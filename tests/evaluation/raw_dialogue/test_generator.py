import pytest

from retracemem.evaluation.raw_dialogue.generator import SyntheticDialogueGenerator
from retracemem.evaluation.raw_dialogue.contracts import DialogueExtractionTarget


def test_generator_determinism():
    gen1 = SyntheticDialogueGenerator(seed=100)
    gen2 = SyntheticDialogueGenerator(seed=100)

    ep1 = gen1.generate_episode("ep_test")
    ep2 = gen2.generate_episode("ep_test")

    assert ep1["example_id"] == ep2["example_id"]
    assert ep1["raw_dialogue"] == ep2["raw_dialogue"]
    assert ep1["gold_graph"] == ep2["gold_graph"]
    assert ep1["gold_actions"] == ep2["gold_actions"]
    assert ep1["gold_final_statuses"] == ep2["gold_final_statuses"]


def test_generator_randomness_across_seeds():
    gen1 = SyntheticDialogueGenerator(seed=10)
    gen2 = SyntheticDialogueGenerator(seed=20)

    ep1 = gen1.generate_episode("ep_test")
    ep2 = gen2.generate_episode("ep_test")

    # Due to different templates and values, they should differ
    assert ep1["raw_dialogue"] != ep2["raw_dialogue"] or ep1["gold_actions"] != ep2["gold_actions"]


def test_generator_valid_dpa_execution():
    gen = SyntheticDialogueGenerator(seed=42)
    ep = gen.generate_episode("ep_test")

    # The statuses should be computed successfully and contain keys
    assert ep["gold_final_statuses"]
    for belief_id in ep["gold_final_statuses"]:
        assert ep["gold_final_statuses"][belief_id] in ("AUTHORIZED", "BLOCKED", "SUPERSEDED", "UNRESOLVED")


def test_generator_validation_against_target_contract():
    gen = SyntheticDialogueGenerator(seed=999)
    ep = gen.generate_episode("ep_test")

    # Build dialogue target
    raw_dialogue_lines = ep["raw_dialogue"].strip().split("\n")
    utterances = []
    for line in raw_dialogue_lines:
        if ":" in line:
            speaker, text = line.split(":", 1)
            speaker = speaker.strip()
            text = text.strip()
        else:
            speaker = "unknown"
            text = line.strip()
        utterances.append({
            "speaker": speaker,
            "text": text
        })

    target_dict = {
        "example_id": "ep_test",
        "dialogue": {
            "utterances": utterances
        },
        "subagent_roles": ep["subagent_roles"],
        "gold_graph": ep["gold_graph"],
        "metadata": {
            "new_evidence_id": ep["new_evidence_id"],
            "gold_actions": ep["gold_actions"],
            "gold_final_statuses": ep["gold_final_statuses"]
        }
    }

    target = DialogueExtractionTarget.from_dict(target_dict)
    # This should run without raising RawDialogueValidationError
    target.validate()

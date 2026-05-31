import pytest

from retracemem.evaluation.raw_dialogue.generator import SyntheticDialogueGenerator
from retracemem.evaluation.raw_dialogue.contracts import DialogueExtractionTarget


def test_generator_determinism():
    gen1 = SyntheticDialogueGenerator(seed=100)
    gen2 = SyntheticDialogueGenerator(seed=100)

    ep1 = gen1.generate_episode("ep_test", case_family="supersedes_basic")
    ep2 = gen2.generate_episode("ep_test", case_family="supersedes_basic")

    assert ep1["example_id"] == ep2["example_id"]
    assert ep1["raw_dialogue"] == ep2["raw_dialogue"]
    assert ep1["gold_graph"] == ep2["gold_graph"]
    assert ep1["gold_actions"] == ep2["gold_actions"]
    assert ep1["gold_final_statuses"] == ep2["gold_final_statuses"]


def test_generator_all_case_families():
    gen = SyntheticDialogueGenerator(seed=42)
    for family in gen.case_families:
        ep = gen.generate_episode(f"ep_{family}", case_family=family)
        assert ep["example_id"] == f"ep_{family}"
        assert ep["case_family"] == family
        assert len(ep["gold_graph"]["evidence_nodes"]) > 0
        assert len(ep["gold_actions"]) > 0
        
        # Check validation target contract
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
            "example_id": ep["example_id"],
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
        target.validate()

        # Check special cases size
        if family == "supersedes_blocks_multi":
            assert len(ep["gold_actions"]) == 2
        elif family == "blocks_then_releases_temporal":
            assert len(ep["gold_actions"]) == 2

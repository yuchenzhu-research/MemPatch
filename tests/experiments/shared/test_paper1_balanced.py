"""Tests for the deterministic ``paper1_balanced`` internal validation suite."""
from __future__ import annotations

from collections import Counter

import pytest

from retracemem.evaluation.multiagent.data.paper1_balanced import (
    generate_paper1_balanced_episodes,
    DOMAINS,
    FAILURE_TYPES,
    VARIANTS,
    GENERATOR_VERSION,
    DATASET_NAME,
)
from retracemem.evaluation.multiagent.cases import (
    load_eval_cases,
    rename_episode_and_gold,
    DATASET_PAPER1_BALANCED,
    DATASET_DEV_EXPANSION,
)
from retracemem.evaluation.multiagent.pipeline import run_retrace_variant_on_episode
from retracemem.proposers.typed_revision_policy import ClosedAPIZeroShotConstrainedProposer


EXPECTED_SIZE = len(FAILURE_TYPES) * len(DOMAINS) * VARIANTS


def test_paper1_balanced_size_is_420() -> None:
    episodes = generate_paper1_balanced_episodes()
    assert EXPECTED_SIZE == 420
    assert len(episodes) == 420


def test_all_14_failure_types_present_and_balanced() -> None:
    episodes = generate_paper1_balanced_episodes()
    counts = Counter(ep.failure_type_public_or_controlled for ep, _ in episodes)
    assert set(counts) == set(FAILURE_TYPES)
    assert len(FAILURE_TYPES) == 14
    # balanced: each failure type appears domains x variants times
    for ftype in FAILURE_TYPES:
        assert counts[ftype] == len(DOMAINS) * VARIANTS


def test_both_domains_present_and_balanced() -> None:
    episodes = generate_paper1_balanced_episodes()
    counts = Counter(ep.domain for ep, _ in episodes)
    assert set(counts) == set(DOMAINS)
    assert set(DOMAINS) == {"software_engineering", "research_workflow"}
    for domain in DOMAINS:
        assert counts[domain] == len(FAILURE_TYPES) * VARIANTS


def test_no_duplicate_episode_ids() -> None:
    episodes = generate_paper1_balanced_episodes()
    ids = [ep.episode_id for ep, _ in episodes]
    assert len(set(ids)) == len(ids)


def test_every_case_has_executable_gold() -> None:
    episodes = generate_paper1_balanced_episodes()
    for ep, gold in episodes:
        assert gold.episode_id == ep.episode_id
        assert gold.failure_type == ep.failure_type_public_or_controlled
        assert gold.gold_snapshot.belief_statuses, ep.episode_id
        assert len(gold.gold_typed_targets) >= 1, ep.episode_id
        sub_ids = {s.submission_id for s in ep.submissions}
        for t in gold.gold_typed_targets:
            assert t.submission_id in sub_ids
            if t.action_type == "SUPERSEDES":
                assert t.target_belief_id is not None
                assert t.replacement_belief_id is not None
            elif t.action_type in ("BLOCKS", "RELEASES"):
                assert t.target_condition_id is not None
            elif t.action_type in ("UNCERTAIN", "REAFFIRMS"):
                assert t.target_belief_id is not None
            elif t.action_type == "NO_REVISION":
                assert t.target_belief_id is None
                assert t.target_condition_id is None
                assert t.replacement_belief_id is None
            # evidence grounding: target evidence must be visible in its submission
            sub = next(s for s in ep.submissions if s.submission_id == t.submission_id)
            visible_ev = {e.evidence_id for e in sub.evidence_context}
            for eid in t.evidence_ids:
                assert eid in visible_ev, (ep.episode_id, eid)


def test_structural_checklist_passes_for_all_cases() -> None:
    episodes = generate_paper1_balanced_episodes()
    for ep, _ in episodes:
        status = ep.metadata.get("semantic_validation_status", {})
        assert status.get("passes_structural_checks") is True, ep.episode_id


def test_every_case_routes_through_dpa_mock_path() -> None:
    """Each gold target set must deterministically reproduce the gold snapshot."""
    proposer = ClosedAPIZeroShotConstrainedProposer(provider_kind="mock", model_id=None, client=None)
    episodes = generate_paper1_balanced_episodes()
    for ep, gold in episodes:
        _raw, _parsed, final_statuses, _trace = run_retrace_variant_on_episode(
            ep, gold, proposer, True
        )
        assert final_statuses == dict(gold.gold_snapshot.belief_statuses), ep.episode_id


def test_new_failure_types_have_expected_actions() -> None:
    episodes = {ep.episode_id: (ep, gold) for ep, gold in generate_paper1_balanced_episodes()}

    # multi-action supersedes+blocks: one submission carries two non-NO_REVISION actions
    ep, gold = episodes["ep_paper1_software_engineering_multi_action_supersedes_blocks_v1"]
    actions = {t.action_type for t in gold.gold_typed_targets}
    assert {"SUPERSEDES", "BLOCKS"} <= actions
    assert gold.requires_multi_action is True

    # multi-action supersedes+releases: blocks then supersedes+releases (3 submissions)
    ep, gold = episodes["ep_paper1_research_workflow_multi_action_supersedes_releases_v1"]
    actions = {t.action_type for t in gold.gold_typed_targets}
    assert {"SUPERSEDES", "RELEASES", "BLOCKS"} <= actions
    assert len(ep.submissions) == 3
    assert gold.requires_multi_action is True

    # blocks_uncertain
    ep, gold = episodes["ep_paper1_software_engineering_blocks_uncertain_v1"]
    actions = {t.action_type for t in gold.gold_typed_targets}
    assert {"BLOCKS", "UNCERTAIN"} <= actions

    # reaffirms_only
    ep, gold = episodes["ep_paper1_software_engineering_reaffirms_only_v1"]
    assert [t.action_type for t in gold.gold_typed_targets] == ["REAFFIRMS"]

    # no_revision
    ep, gold = episodes["ep_paper1_research_workflow_no_revision_v1"]
    assert [t.action_type for t in gold.gold_typed_targets] == ["NO_REVISION"]


def test_metadata_marks_internal_validation_not_training() -> None:
    episodes = generate_paper1_balanced_episodes()
    for ep, gold in episodes:
        meta = ep.metadata
        assert meta["dataset_name"] == DATASET_NAME
        assert meta["generator_version"] == GENERATOR_VERSION
        assert meta["training_eligible"] is False
        assert meta["scientific_status"] == "internal_balanced_validation"
        assert ep.split == "internal_balanced_validation"
        assert gold.metadata == meta


def test_heldout_base_rename_remains_compatible() -> None:
    """The legacy namespace rename helper must still work on a paper1 case."""
    ep, gold = generate_paper1_balanced_episodes()[0]
    renamed_ep, renamed_gold = rename_episode_and_gold(ep, gold)
    assert renamed_ep.episode_id.endswith("__heldout_base")
    assert renamed_gold.episode_id == renamed_ep.episode_id
    # gold belief ids are renamed consistently
    assert all("__heldout_base" in bid for bid in renamed_gold.gold_snapshot.belief_statuses)


def test_load_eval_cases_dataset_selection() -> None:
    dev = load_eval_cases(dataset=DATASET_DEV_EXPANSION)
    assert len(dev) == 70

    paper1 = load_eval_cases(dataset=DATASET_PAPER1_BALANCED)
    assert len(paper1) == 420

    limited = load_eval_cases(max_cases=10, dataset=DATASET_PAPER1_BALANCED)
    assert len(limited) == 10

    # default stays dev_expansion
    assert len(load_eval_cases()) == 70


def test_load_eval_cases_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError):
        load_eval_cases(dataset="does_not_exist")

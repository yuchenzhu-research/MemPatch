from __future__ import annotations

import pytest

from retracemem.evaluation.multiagent.data.dev_expansion import generate_expanded_episodes
from retracemem.multiagent.utils import build_candidate_actions, detect_local_conflict
from retracemem.proposers.typed_revision_policy import (
    ClosedAPIZeroShotConstrainedProposer,
    ConflictAwareConstrainedProposer,
    CONFLICT_HANDLING_RULE,
)


@pytest.fixture(scope="module")
def episodes_by_id():
    return {ep.episode_id: (ep, gold) for ep, gold in generate_expanded_episodes()}


def _conflict_submission(episodes_by_id):
    # The reviewer submission (sub2) carries the competing belief.
    ep, _ = episodes_by_id["ep_expansion_research_workflow_cross_agent_conflict_v1"]
    return ep.submissions[1]


# ---------------------------------------------------------------------------
# 1. Cross-agent conflict candidate affordance
# ---------------------------------------------------------------------------

def test_cross_agent_conflict_exposes_conflict_resolving_affordance(episodes_by_id):
    sub = _conflict_submission(episodes_by_id)
    candidates = build_candidate_actions(sub)
    action_types = {c["action_type"] for c in candidates}

    # At least one conflict-resolving option is available.
    assert action_types & {"UNCERTAIN", "BLOCKS", "SUPERSEDES"}
    # Specifically UNCERTAIN here, since there is no replacement/condition handle.
    assert "UNCERTAIN" in action_types

    # REAFFIRMS must not be the only non-NO_REVISION action available.
    non_no_revision = action_types - {"NO_REVISION"}
    assert non_no_revision != {"REAFFIRMS"}
    assert "UNCERTAIN" in non_no_revision


def test_detect_local_conflict_triggers_only_on_cross_agent_conflict(episodes_by_id):
    triggered_failure_types = {}
    for ep, _ in episodes_by_id.values():
        for sub in ep.submissions:
            ok, established, new = detect_local_conflict(sub)
            if ok:
                ft = ep.failure_type_public_or_controlled
                triggered_failure_types[ft] = triggered_failure_types.get(ft, 0) + 1
                # When triggered there is a clear established-vs-new split.
                assert established and new
                assert not set(established) & set(new)

    # The detector is surgical: it fires only on cross_agent_conflict sub2.
    assert set(triggered_failure_types) == {"cross_agent_conflict"}


def test_detect_local_conflict_does_not_trigger_on_scope_expansion(episodes_by_id):
    ep, _ = episodes_by_id["ep_expansion_software_engineering_scope_expansion_v1"]
    for sub in ep.submissions:
        ok, _, _ = detect_local_conflict(sub)
        assert ok is False


# ---------------------------------------------------------------------------
# 2. Prompt rule
# ---------------------------------------------------------------------------

def test_conflict_aware_system_prompt_contains_conflict_rule(episodes_by_id):
    sub = _conflict_submission(episodes_by_id)
    candidates = build_candidate_actions(sub)
    proposer = ConflictAwareConstrainedProposer()
    system_prompt = proposer.build_system_prompt(candidates)

    assert "Conflict handling rule" in system_prompt
    # Distinguishes REAFFIRMS from the conflict/invalidation/replacement actions.
    assert "REAFFIRMS" in system_prompt
    assert "UNCERTAIN" in system_prompt
    assert "BLOCKS" in system_prompt
    assert "SUPERSEDES" in system_prompt
    assert "do not independently reaffirm both" in system_prompt


def test_conflict_aware_propose_emits_conflict_notice_and_audit(episodes_by_id):
    sub = _conflict_submission(episodes_by_id)
    proposer = ConflictAwareConstrainedProposer()  # mock mode (no client)
    out = proposer.propose(sub)

    assert out.policy_variant == "zero_shot_constrained_conflict_aware"
    assert out.metadata.get("conflict_warning_triggered") is True
    assert out.metadata.get("prompt_variant") == "zero_shot_constrained_conflict_aware"
    assert out.metadata.get("conflict_established_belief_ids")
    assert out.metadata.get("conflict_new_belief_ids")

    prompt = out.metadata["prompt"]
    assert "### Conflict handling rule" in prompt
    assert "### Conflict Notice" in prompt
    assert "conflict-resolving: belief" in prompt


def test_conflict_aware_no_notice_when_no_conflict(episodes_by_id):
    # duplicate_evidence has a single candidate belief -> no local conflict.
    ep, _ = episodes_by_id["ep_expansion_software_engineering_duplicate_evidence_v1"]
    sub = ep.submissions[1]
    proposer = ConflictAwareConstrainedProposer()
    out = proposer.propose(sub)

    assert out.metadata.get("conflict_warning_triggered") is False
    prompt = out.metadata["prompt"]
    # Stable rule is always present, but the per-case notice/annotation is not.
    assert "### Conflict handling rule" in prompt
    assert "### Conflict Notice" not in prompt
    assert "conflict-resolving: belief" not in prompt


# ---------------------------------------------------------------------------
# 3. No-regression: default constrained behavior is untouched
# ---------------------------------------------------------------------------

def test_default_constrained_prompt_has_no_conflict_sections(episodes_by_id):
    sub = _conflict_submission(episodes_by_id)
    proposer = ClosedAPIZeroShotConstrainedProposer()
    out = proposer.propose(sub)

    assert out.policy_variant == "zero_shot_constrained"
    assert "conflict_warning_triggered" not in out.metadata
    prompt = out.metadata["prompt"]
    assert "Conflict handling rule" not in prompt
    assert "Conflict Notice" not in prompt


def _affordances_for(episodes_by_id, episode_id):
    ep, _ = episodes_by_id[episode_id]
    per_sub = []
    for sub in ep.submissions:
        per_sub.append({c["action_type"] for c in build_candidate_actions(sub)})
    return per_sub


def test_no_regression_direct_supersession_exposes_supersedes(episodes_by_id):
    per_sub = _affordances_for(episodes_by_id, "ep_expansion_software_engineering_direct_supersession_v1")
    assert any("SUPERSEDES" in actions for actions in per_sub)


def test_no_regression_stale_propagation_exposes_blocks(episodes_by_id):
    per_sub = _affordances_for(episodes_by_id, "ep_expansion_software_engineering_stale_propagation_v1")
    assert any("BLOCKS" in actions for actions in per_sub)


def test_no_regression_temporary_blocker_recovery_exposes_blocks_and_releases(episodes_by_id):
    per_sub = _affordances_for(
        episodes_by_id, "ep_expansion_research_workflow_temporary_blocker_recovery_v1"
    )
    assert any("BLOCKS" in actions for actions in per_sub)
    assert any("RELEASES" in actions for actions in per_sub)


def test_no_regression_duplicate_evidence_allows_no_revision_and_reaffirms(episodes_by_id):
    per_sub = _affordances_for(episodes_by_id, "ep_expansion_software_engineering_duplicate_evidence_v1")
    assert any("NO_REVISION" in actions for actions in per_sub)
    assert any("REAFFIRMS" in actions for actions in per_sub)


def test_conflict_handling_rule_is_stable_text():
    # Guards the stable prompt section against accidental edits.
    assert "Conflict handling rule" in CONFLICT_HANDLING_RULE
    assert "do not independently reaffirm both" in CONFLICT_HANDLING_RULE

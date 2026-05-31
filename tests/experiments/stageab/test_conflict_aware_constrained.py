"""Tests for the conflict-aware constrained Stage A variant.

These cover Round 3 requirements: conflict-resolving affordances are available,
the conflict-aware prompt carries the conflict-handling rule and distinguishes
REAFFIRMS from UNCERTAIN/BLOCKS/SUPERSEDES, no regression to the existing
constrained policy or to the other failure types' affordances, and the
structural conflict trigger fires only on competing-belief shapes.

All tests are offline (no live API).
"""
from __future__ import annotations

import json

import pytest

from retracemem.evaluation.multiagent.data.dev_expansion import generate_expanded_episodes
from retracemem.multiagent.utils import build_candidate_actions, detect_competing_beliefs
from retracemem.proposers.typed_revision_policy import ClosedAPIZeroShotConstrainedProposer


def _episodes_by_failure_type():
    """Map failure_type -> (episode, gold) using the public structural attribute.

    We never hard-code episode IDs; we read the episode's declared failure type.
    """
    out: dict[str, tuple] = {}
    for ep, gold in generate_expanded_episodes():
        out.setdefault(ep.failure_type_public_or_controlled, (ep, gold))
    return out


@pytest.fixture(scope="module")
def episodes():
    return _episodes_by_failure_type()


@pytest.fixture
def conflict_submission(episodes):
    ep, _ = episodes["cross_agent_conflict"]
    # The reviewer submission carries both competing beliefs.
    return ep.submissions[-1]


# --- 1. Cross-agent conflict candidate affordances ---------------------------

def test_conflict_case_exposes_a_conflict_resolving_affordance(conflict_submission):
    candidates = build_candidate_actions(conflict_submission)
    types = {c["action_type"] for c in candidates}
    # At least one conflict-resolving option must be present.
    assert types & {"UNCERTAIN", "BLOCKS", "SUPERSEDES"}, types
    # Specifically UNCERTAIN must be available for the competing beliefs.
    assert "UNCERTAIN" in types


def test_conflict_case_reaffirms_is_not_the_only_revision_action(conflict_submission):
    candidates = build_candidate_actions(conflict_submission)
    non_no_revision = {
        c["action_type"] for c in candidates if c["action_type"] != "NO_REVISION"
    }
    assert "REAFFIRMS" in non_no_revision
    # REAFFIRMS must not be the sole non-NO_REVISION affordance.
    assert non_no_revision - {"REAFFIRMS"}, non_no_revision


# --- 2. Prompt rule ----------------------------------------------------------

def test_conflict_aware_prompt_contains_conflict_handling_rule(conflict_submission):
    candidates = build_candidate_actions(conflict_submission)
    prompt = ClosedAPIZeroShotConstrainedProposer(conflict_aware=True).build_system_prompt(candidates)
    assert "Conflict handling rule" in prompt


def test_conflict_aware_prompt_distinguishes_reaffirms_from_alternatives(conflict_submission):
    candidates = build_candidate_actions(conflict_submission)
    prompt = ClosedAPIZeroShotConstrainedProposer(conflict_aware=True).build_system_prompt(candidates)
    assert "REAFFIRMS" in prompt
    assert "UNCERTAIN" in prompt
    # The rule references at least one structural alternative beyond UNCERTAIN.
    assert ("BLOCKS" in prompt) or ("SUPERSEDES" in prompt)


def test_conflict_aware_user_prompt_emits_conflict_notice(conflict_submission):
    candidates = build_candidate_actions(conflict_submission)
    proposer = ClosedAPIZeroShotConstrainedProposer(conflict_aware=True)
    user_prompt = proposer.build_user_prompt(conflict_submission, candidates)
    assert "Conflict Notice" in user_prompt
    # The competing belief IDs are surfaced (method-visible, not gold).
    for b in conflict_submission.candidate_beliefs:
        assert b.belief_id in user_prompt


# --- Default constrained policy must be unchanged ----------------------------

def test_default_constrained_prompt_has_no_conflict_rule(conflict_submission):
    candidates = build_candidate_actions(conflict_submission)
    base = ClosedAPIZeroShotConstrainedProposer()  # conflict_aware defaults to False
    prompt = base.build_system_prompt(candidates)
    assert "Conflict handling rule" not in prompt
    assert base.policy_variant == "zero_shot_constrained"


def test_default_constrained_user_prompt_has_no_conflict_notice(conflict_submission):
    candidates = build_candidate_actions(conflict_submission)
    base = ClosedAPIZeroShotConstrainedProposer()
    user_prompt = base.build_user_prompt(conflict_submission, candidates)
    assert "Conflict Notice" not in user_prompt


def test_conflict_aware_has_distinct_policy_variant_label():
    assert (
        ClosedAPIZeroShotConstrainedProposer(conflict_aware=True).policy_variant
        == "zero_shot_constrained_conflict_aware"
    )


# --- Decision audit (debug only, does not change DPA edges) ------------------

def test_conflict_warning_triggered_recorded_in_metadata(conflict_submission):
    candidates = build_candidate_actions(conflict_submission)
    proposer = ClosedAPIZeroShotConstrainedProposer(conflict_aware=True)
    reaffirms_ids = [c["candidate_action_id"] for c in candidates if c["action_type"] == "REAFFIRMS"]
    response = json.dumps({"selected_candidate_action_ids": reaffirms_ids, "rejection_reasons": {}})
    out = proposer.parse_response(
        response, example_id="ex_test", submission=conflict_submission, candidates=candidates
    )
    assert out.parsing_valid is True
    assert out.metadata.get("prompt_variant") == "conflict_aware"
    assert out.metadata.get("conflict_warning_triggered") is True


def test_default_constrained_does_not_emit_conflict_metadata(conflict_submission):
    candidates = build_candidate_actions(conflict_submission)
    base = ClosedAPIZeroShotConstrainedProposer()
    reaffirms_ids = [c["candidate_action_id"] for c in candidates if c["action_type"] == "REAFFIRMS"]
    response = json.dumps({"selected_candidate_action_ids": reaffirms_ids, "rejection_reasons": {}})
    out = base.parse_response(
        response, example_id="ex_test", submission=conflict_submission, candidates=candidates
    )
    assert "conflict_warning_triggered" not in out.metadata
    assert "prompt_variant" not in out.metadata


# --- 3. No-regression for the other failure types ----------------------------

def test_direct_supersession_still_exposes_supersedes(episodes):
    ep, _ = episodes["direct_supersession"]
    candidates = build_candidate_actions(ep.submissions[-1])
    assert any(c["action_type"] == "SUPERSEDES" for c in candidates)


def test_stale_propagation_still_exposes_blocks(episodes):
    ep, _ = episodes["stale_propagation"]
    # Aggregate affordances across the submissions that carry conditions.
    types = set()
    for sub in ep.submissions:
        types.update(c["action_type"] for c in build_candidate_actions(sub))
    assert "BLOCKS" in types


def test_temporary_blocker_recovery_still_exposes_blocks_and_releases(episodes):
    ep, _ = episodes["temporary_blocker_recovery"]
    types = set()
    for sub in ep.submissions:
        types.update(c["action_type"] for c in build_candidate_actions(sub))
    assert "BLOCKS" in types
    assert "RELEASES" in types


def test_duplicate_evidence_allows_no_revision_and_reaffirms(episodes):
    ep, _ = episodes["duplicate_evidence"]
    candidates = build_candidate_actions(ep.submissions[-1])
    types = {c["action_type"] for c in candidates}
    assert "NO_REVISION" in types
    assert "REAFFIRMS" in types


# --- Structural trigger only fires on competing-belief shapes ----------------

def test_conflict_trigger_fires_only_on_competing_belief_shape(episodes):
    fires = {
        ftype: detect_competing_beliefs(ep.submissions[-1])
        for ftype, (ep, _) in episodes.items()
    }
    assert fires["cross_agent_conflict"] is True
    # Must NOT misfire on shapes that carry conditions or replacements.
    assert fires["scope_expansion"] is False
    assert fires["direct_supersession"] is False
    assert fires["duplicate_evidence"] is False
    assert fires["temporary_blocker_recovery"] is False
    assert fires["ambiguous_update"] is False

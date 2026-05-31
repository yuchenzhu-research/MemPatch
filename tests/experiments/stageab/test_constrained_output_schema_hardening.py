"""Schema-hardening tests for the constrained Stage A proposers and parser.

Covers the conflict-aware schema fix:
- minimal constrained output schema is requested (no rejection_reasons),
- conflict-aware prompt repeats the minimal-schema reminder at the very end,
- the JSON parser fails closed on a truncated outer object instead of falling
  back to a nested ``rejection_reasons`` object (which previously produced the
  misleading "missing required key" error),
- both constrained variants still parse the minimal valid object.
"""
from __future__ import annotations

import json

import pytest

from retracemem.multiagent.parser import (
    extract_json_object,
    StructuredParseError,
    ParseErrorCode,
)
from retracemem.multiagent.utils import build_candidate_actions
from retracemem.schemas import (
    EvidenceNode,
    BeliefNode,
    ConditionNode,
    DependencyEdge,
)
from retracemem.evaluation.multiagent.contracts import FixedCandidateSubmission
from retracemem.evaluation.multiagent.data.dev_expansion import generate_expanded_episodes
from retracemem.proposers.typed_revision_policy import (
    ClosedAPIZeroShotConstrainedProposer,
    ConflictAwareConstrainedProposer,
    CONFLICT_AWARE_FINAL_SCHEMA_REMINDER,
)


@pytest.fixture
def tiny_submission():
    ev = EvidenceNode("ev_1", "sess_1", "2026-05-30T00:00:00Z", "Some evidence", "dataset", "pointer")
    b = BeliefNode("b_1", "Proposition 1", ("ev_1",))
    b2 = BeliefNode("b_2", "Proposition 2", ("ev_1",))
    c = ConditionNode("c_1", "scope_1", "Condition 1")
    dep = DependencyEdge("dep_1", "b_1", "c_1", "system")
    return FixedCandidateSubmission(
        submission_id="sub_1",
        producer_id="writer",
        producer_role="writer",
        task_id="task_1",
        parent_snapshot_id="snapshot_init",
        observed_at="2026-05-30T00:00:00Z",
        instance_id="inst_1",
        query_id="q_1",
        query="Check status?",
        evidence_context=(ev,),
        new_evidence_id="ev_1",
        candidate_beliefs=(b,),
        candidate_replacement_beliefs=(b2,),
        candidate_conditions_by_belief=(("b_1", (c,)),),
        dependency_edges_by_belief=(("b_1", (dep,)),),
    )


@pytest.fixture(scope="module")
def episodes_by_id():
    return {ep.episode_id: (ep, gold) for ep, gold in generate_expanded_episodes()}


def _conflict_submission(episodes_by_id):
    ep, _ = episodes_by_id["ep_expansion_research_workflow_cross_agent_conflict_v1"]
    return ep.submissions[1]


# ---------------------------------------------------------------------------
# 1. Minimal constrained output schema in the prompt
# ---------------------------------------------------------------------------

def test_default_constrained_prompt_requests_minimal_schema(tiny_submission):
    proposer = ClosedAPIZeroShotConstrainedProposer()
    candidates = build_candidate_actions(tiny_submission)
    system_prompt = proposer.build_system_prompt(candidates)

    assert "selected_candidate_action_ids" in system_prompt
    # The non-diagnostic constrained schema must not ask for rejection_reasons.
    assert "rejection_reasons" not in system_prompt.split("### Output Format")[1] \
        or "Do NOT include rejection_reasons" in system_prompt
    assert "Do NOT include rejection_reasons" in system_prompt


def test_conflict_aware_prompt_contains_minimal_schema_reminder(episodes_by_id):
    sub = _conflict_submission(episodes_by_id)
    proposer = ConflictAwareConstrainedProposer()  # mock mode
    out = proposer.propose(sub)
    prompt = out.metadata["prompt"]

    # The minimal-schema reminder is present and is the final schema instruction.
    assert "Final output schema reminder" in prompt
    assert "exactly one required key: selected_candidate_action_ids" in prompt
    # The reminder is the last reminder block in the prompt.
    assert prompt.rstrip().endswith(CONFLICT_AWARE_FINAL_SCHEMA_REMINDER.rstrip())


def test_conflict_aware_prompt_forbids_rejection_reasons_and_full_actions(episodes_by_id):
    sub = _conflict_submission(episodes_by_id)
    proposer = ConflictAwareConstrainedProposer()
    out = proposer.propose(sub)
    prompt = out.metadata["prompt"]

    # Forbidden output content for live constrained outputs.
    assert "Do not include rejection_reasons" in prompt
    assert "full actions" in prompt
    assert "prose" in prompt


# ---------------------------------------------------------------------------
# 2. Parser fail-closed on truncated outer JSON with nested rejection_reasons
# ---------------------------------------------------------------------------

# Outer object never closes (truncated), but the nested rejection_reasons object
# is well-formed. This is the exact failure shape observed in the live run.
_TRUNCATED_RESPONSE = (
    "{\n"
    '  "selected_candidate_action_ids": ["act_no_revision"],\n'
    '  "rejection_reasons": {\n'
    '    "act_blocks_c_1": "would invalidate a still-valid condition",\n'
    '    "act_supersedes_b_1": "no grounded replacement in evidence"\n'
    "  }"
    # NOTE: missing final closing brace of the outer object.
)


def test_extract_json_object_rejects_truncated_outer_with_nested_object():
    # Without the guard, the scanner returns the nested rejection_reasons dict.
    fallback = extract_json_object(_TRUNCATED_RESPONSE)
    assert "selected_candidate_action_ids" not in fallback  # documents old behavior

    # With the guard, parsing fails closed rather than returning a nested object.
    with pytest.raises(StructuredParseError) as exc_info:
        extract_json_object(
            _TRUNCATED_RESPONSE,
            require_top_level_keys={"selected_candidate_action_ids"},
        )
    err = exc_info.value
    assert err.code == ParseErrorCode.JSON_DECODE_FAILED
    assert "complete top-level constrained JSON object" in err.message


def test_extract_json_object_require_keys_returns_complete_object():
    good = '{"selected_candidate_action_ids": ["act_no_revision"], "rejection_reasons": {}}'
    obj = extract_json_object(good, require_top_level_keys={"selected_candidate_action_ids"})
    assert obj["selected_candidate_action_ids"] == ["act_no_revision"]


def test_constrained_parse_response_reports_clear_error_on_truncated_outer(tiny_submission):
    proposer = ClosedAPIZeroShotConstrainedProposer()
    candidates = build_candidate_actions(tiny_submission)
    out = proposer.parse_response(
        _TRUNCATED_RESPONSE,
        example_id="ex_test",
        submission=tiny_submission,
        candidates=candidates,
    )
    assert out.parsing_valid is False
    joined = "\n".join(out.errors)
    # The error is now about the malformed top-level object, NOT a misleading
    # "missing required key" message when the key is actually present in raw text.
    assert "complete top-level constrained JSON object" in joined
    assert "missing required key 'selected_candidate_action_ids'" not in joined


# ---------------------------------------------------------------------------
# 3. Both constrained variants still parse the minimal valid object
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "proposer_cls",
    [ClosedAPIZeroShotConstrainedProposer, ConflictAwareConstrainedProposer],
)
def test_minimal_valid_object_parses_for_both_variants(tiny_submission, proposer_cls):
    proposer = proposer_cls()
    candidates = build_candidate_actions(tiny_submission)
    minimal = json.dumps({"selected_candidate_action_ids": ["act_no_revision"]})
    out = proposer.parse_response(
        minimal,
        example_id="ex_test",
        submission=tiny_submission,
        candidates=candidates,
    )
    assert out.parsing_valid is True
    assert len(out.parsed_actions) == 1
    assert out.parsed_actions[0].action_type == "NO_REVISION"


# ---------------------------------------------------------------------------
# 4. Conflict metadata is surfaced into Stage A parsed artifacts
# ---------------------------------------------------------------------------

def test_conflict_metadata_surfaced_into_parsed_artifacts(episodes_by_id):
    from retracemem.evaluation.multiagent.pipeline import run_retrace_variant_on_episode

    ep, gold = episodes_by_id["ep_expansion_research_workflow_cross_agent_conflict_v1"]
    proposer = ConflictAwareConstrainedProposer()  # no client -> internal mock
    # mock=False so the real proposer runs (mock=True would bypass it with gold).
    _raw, parsed, _statuses, _trace = run_retrace_variant_on_episode(
        ep, gold, proposer, mock=False
    )

    # The conflict-bearing reviewer submission must carry the conflict metadata.
    by_sub = {row["submission_id"]: row for row in parsed}
    triggered_rows = [r for r in parsed if r.get("conflict_warning_triggered")]
    assert triggered_rows, "expected at least one conflict-triggered submission"
    for row in parsed:
        assert row["prompt_variant"] == "zero_shot_constrained_conflict_aware"
    for row in triggered_rows:
        assert row["conflict_established_belief_ids"]
        assert row["conflict_new_belief_ids"]
    assert by_sub  # sanity


def test_extra_rejection_reasons_key_is_still_tolerated(tiny_submission):
    # Backward compatibility: a well-formed object with an extra rejection_reasons
    # key still parses (the parser only requires selected_candidate_action_ids).
    proposer = ClosedAPIZeroShotConstrainedProposer()
    candidates = build_candidate_actions(tiny_submission)
    resp = json.dumps({
        "selected_candidate_action_ids": ["act_no_revision"],
        "rejection_reasons": {"act_blocks_c_1": "n/a"},
    })
    out = proposer.parse_response(
        resp, example_id="ex_test", submission=tiny_submission, candidates=candidates
    )
    assert out.parsing_valid is True

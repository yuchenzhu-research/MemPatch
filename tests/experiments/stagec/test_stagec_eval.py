from __future__ import annotations

import json

import pytest

from retracemem.schemas import (
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceNode,
)
from experiments.multiagent.contracts import (
    FixedCandidateGoldRecord,
    FixedCandidateInputEpisode,
    FixedCandidateSubmission,
    GoldSnapshotExpectation,
    TypedRevisionTarget,
)
from experiments.multiagent.stagec_adapter_proposer import (
    DirectoryGenerationSource,
    LocalAdapterReplayProposer,
    MappingGenerationSource,
    build_replay_proposer,
    no_revision_generation,
)
from experiments.multiagent.run_stagec_eval import (
    StageCEvalConfig,
    build_case_record,
    run_stagec_eval,
)


# ---------------------------------------------------------------------------
# Minimal in-memory episode/gold fixture (mirrors the Stage A/B test style).
# ---------------------------------------------------------------------------
@pytest.fixture
def supersession_episode_and_gold():
    ev_old = EvidenceNode("ev_old", "sess_1", "2026-05-01T00:00:00Z", "Old fact", "dataset", "ptr")
    ev_new = EvidenceNode("ev_new", "sess_1", "2026-05-30T00:00:00Z", "Updated fact", "dataset", "ptr")
    b_old = BeliefNode("b_old", "Service runs on port 8080", ("ev_old",))
    b_new = BeliefNode("b_new", "Service runs on port 9090", ("ev_new",))
    cond = ConditionNode("c_1", "scope_1", "Deployment active")
    dep = DependencyEdge("dep_1", "b_old", "c_1", "system")

    sub = FixedCandidateSubmission(
        submission_id="sub_1",
        producer_id="writer",
        producer_role="writer",
        task_id="task_1",
        parent_snapshot_id="snap_init",
        observed_at="2026-05-30T00:00:00Z",
        instance_id="inst_1",
        query_id="q_1",
        query="Which port does the service use?",
        evidence_context=(ev_old, ev_new),
        new_evidence_id="ev_new",
        candidate_beliefs=(b_old,),
        candidate_replacement_beliefs=(b_new,),
        candidate_conditions_by_belief=(("b_old", (cond,)),),
        dependency_edges_by_belief=(("b_old", (dep,)),),
    )

    episode = FixedCandidateInputEpisode(
        episode_id="ep_test",
        domain="software_engineering",
        failure_type_public_or_controlled="direct_supersession",
        subagent_roles=("writer",),
        submissions=(sub,),
        downstream_tasks=(),
    )

    gold = FixedCandidateGoldRecord(
        episode_id="ep_test",
        gold_snapshot=GoldSnapshotExpectation(
            belief_statuses={"b_old": "SUPERSEDED", "b_new": "AUTHORIZED"}
        ),
        gold_typed_targets=(
            TypedRevisionTarget(
                "sub_1",
                "SUPERSEDES",
                target_belief_id="b_old",
                replacement_belief_id="b_new",
                evidence_ids=("ev_new",),
            ),
        ),
        failure_type="direct_supersession",
    )
    return episode, gold


def _gold_generation_for_submission(sub, gold) -> str:
    """Test-only helper: render gold typed targets as decoded proposer text."""
    targets = [t for t in gold.gold_typed_targets if t.submission_id == sub.submission_id]
    if not targets:
        return no_revision_generation(sub)
    actions = []
    for t in targets:
        ev_ids = list(t.evidence_ids)
        if sub.new_evidence_id not in ev_ids:
            ev_ids.append(sub.new_evidence_id)
        actions.append(
            {
                "action_type": t.action_type,
                "target_belief_id": t.target_belief_id,
                "target_condition_id": t.target_condition_id,
                "replacement_belief_id": t.replacement_belief_id,
                "rationale": t.rationale or "gold",
                "evidence_ids": ev_ids,
            }
        )
    return json.dumps(actions)


# ---------------------------------------------------------------------------
# Proposer-level tests
# ---------------------------------------------------------------------------
def test_mock_generation_is_schema_valid(supersession_episode_and_gold):
    ep, _ = supersession_episode_and_gold
    sub = ep.submissions[0]
    proposer = build_replay_proposer(mock=True)
    out = proposer.propose(sub)

    assert out.parsing_valid is True
    assert out.policy_variant == "mock_smoke"
    assert [a.action_type for a in out.parsed_actions] == ["NO_REVISION"]
    assert out.metadata["raw_response"]
    assert out.metadata["prompt"]
    assert out.metadata["first_pass_valid_json"] is True


def test_replay_valid_supersedes_grounds_and_parses(supersession_episode_and_gold):
    ep, gold = supersession_episode_and_gold
    sub = ep.submissions[0]
    gen = _gold_generation_for_submission(sub, gold)
    proposer = LocalAdapterReplayProposer(
        MappingGenerationSource({"sub_1": gen}),
        policy_variant="adapter_replay",
        backbone_model="local-test-4b",
        checkpoint_id="ckpt-1",
    )
    out = proposer.propose(sub)

    assert out.parsing_valid is True, out.errors
    assert [a.action_type for a in out.parsed_actions] == ["SUPERSEDES"]
    assert out.backbone_model == "local-test-4b"
    assert out.checkpoint_id == "ckpt-1"
    # A SUPERSEDES proposal must produce at least one admitted edge batch.
    assert out.proposal_batches


def test_replay_missing_generation_fails_closed(supersession_episode_and_gold):
    ep, _ = supersession_episode_and_gold
    sub = ep.submissions[0]
    proposer = LocalAdapterReplayProposer(MappingGenerationSource({}))
    out = proposer.propose(sub)

    assert out.parsing_valid is False
    assert out.proposal_batches == ()
    assert out.metadata["failure_reason"] == "missing_generation"
    assert out.errors


def test_replay_malformed_generation_fails_closed(supersession_episode_and_gold):
    ep, _ = supersession_episode_and_gold
    sub = ep.submissions[0]
    proposer = LocalAdapterReplayProposer(
        MappingGenerationSource({"sub_1": "this is not json at all"})
    )
    out = proposer.propose(sub)

    assert out.parsing_valid is False
    assert out.metadata["failure_reason"] == "parse_or_validation_error"
    assert out.errors


def test_directory_generation_source(tmp_path, supersession_episode_and_gold):
    ep, gold = supersession_episode_and_gold
    sub = ep.submissions[0]
    gen_dir = tmp_path / "gens"
    gen_dir.mkdir()
    (gen_dir / "sub_1.txt").write_text(_gold_generation_for_submission(sub, gold), encoding="utf-8")

    source = DirectoryGenerationSource(gen_dir)
    assert source(sub) is not None
    proposer = LocalAdapterReplayProposer(source)
    out = proposer.propose(sub)
    assert out.parsing_valid is True


# ---------------------------------------------------------------------------
# Per-case record test
# ---------------------------------------------------------------------------
def test_build_case_record_has_required_fields(supersession_episode_and_gold):
    from experiments.multiagent.run_stageab_api_eval import run_retrace_variant_on_episode

    ep, gold = supersession_episode_and_gold
    sub = ep.submissions[0]
    gen = _gold_generation_for_submission(sub, gold)
    proposer = LocalAdapterReplayProposer(MappingGenerationSource({"sub_1": gen}))

    raw_a, parsed_a, final_dpa, trace = run_retrace_variant_on_episode(
        ep, gold, proposer, mock=False
    )
    record = build_case_record(ep, gold, raw_a, parsed_a, final_dpa, trace)

    # Requirement-6 fields are all present.
    assert record["episode_id"] == "ep_test"
    assert record["case_id"] == "ep_test"
    sub_rec = record["submissions"][0]
    for key in (
        "method_visible_input",
        "raw_proposer_output",
        "parsed_actions",
        "gold_typed_targets",
        "action_metrics",
        "parse_or_validation_error",
    ):
        assert key in sub_rec
    assert record["final_dpa_belief_statuses"]
    assert record["gold_belief_statuses"] == {"b_old": "SUPERSEDED", "b_new": "AUTHORIZED"}
    assert "correctness" in record

    # Replaying gold actions through the kernel reproduces gold belief statuses.
    assert record["correctness"]["belief_status_accuracy"] == 1.0
    assert record["correctness"]["episode_exact_match"] is True


# ---------------------------------------------------------------------------
# End-to-end runner smoke test (offline, no API).
# ---------------------------------------------------------------------------
def test_run_stagec_eval_smoke_writes_outputs(tmp_path):
    out_dir = tmp_path / "stagec_run"
    config = StageCEvalConfig(
        proposer_source="mock",
        max_cases=2,
        smoke=True,
        output_dir=str(out_dir),
    )
    metrics, manifest = run_stagec_eval(config)

    assert "stage_c" in metrics
    assert "final_status_accuracy" in metrics["stage_c"]
    assert manifest["stage"] == "C"
    assert manifest["is_live_api_result"] is False
    assert manifest["cases_evaluated"] == 2

    for fname in (
        "stagec_raw.jsonl",
        "stagec_parsed.jsonl",
        "dpa_traces.jsonl",
        "stagec_records.jsonl",
        "metrics.json",
        "manifest.json",
        "failure_breakdown.csv",
    ):
        assert (out_dir / fname).exists(), f"missing {fname}"

    records = [
        json.loads(line)
        for line in (out_dir / "stagec_records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(records) == 2
    assert all("correctness" in r for r in records)


def test_run_stagec_eval_replay_requires_generations_dir():
    config = StageCEvalConfig(proposer_source="replay", generations_dir=None)
    with pytest.raises(ValueError):
        run_stagec_eval(config)

"""End-to-end smoke test (Section G).

Subagent A proposes an old belief; B supersedes it; C blocks a condition another
belief requires; D releases that condition. Expected DPA outcome:
  old belief         -> SUPERSEDED
  replacement belief -> AUTHORIZED
  blocked-then-released belief -> AUTHORIZED
plus an audit trace carrying evidence ids, gate decisions, and defeat paths.
"""
from __future__ import annotations

from retrace_learn.data.build_synthetic_raw_dialogue import get_smoke_episode
from retrace_learn.data.export_graph_sft import build_rows as build_graph_rows
from retrace_learn.data.export_revision_sft import build_rows as build_revision_rows
from retrace_learn.data.export_rl_rollouts import build_preference_pairs, build_rows as build_rollout_rows
from retrace_learn.runtime.graph_extractor import RuleBasedGraphExtractor
from retrace_learn.runtime.learned_proposer import ScriptedProposer
from retrace_learn.runtime.dpa_runtime import run_from_text
from retrace_learn.eval.eval_graph_extraction import evaluate as eval_graph
from retrace_learn.eval.eval_revision_policy import evaluate as eval_policy


def test_graph_extractor_recovers_nodes():
    ep = get_smoke_episode()
    g = RuleBasedGraphExtractor().extract(ep.raw_dialogue, ep.subagent_roles)
    assert {e["evidence_id"] for e in g["evidence_nodes"]} == {"ev1", "ev2", "ev3", "ev4"}
    assert {b["belief_id"] for b in g["belief_nodes"]} == {"b_old", "b_dep"}
    assert {b["belief_id"] for b in g["candidate_replacement_beliefs"]} == {"b_new"}
    assert {c["condition_id"] for c in g["condition_nodes"]} == {"c1"}
    assert (g["dependency_edges"][0]["belief_id"], g["dependency_edges"][0]["condition_id"]) == ("b_dep", "c1")


def test_end_to_end_supersede_block_release():
    ep = get_smoke_episode()
    view = ep.build_view()
    proposer = ScriptedProposer(list(ep.gold_actions))
    out = proposer.propose(view)
    assert out.parsing_valid
    # proposer emitted SUPERSEDES, BLOCKS, RELEASES
    assert {a.action_type for a in out.actions} == {"SUPERSEDES", "BLOCKS", "RELEASES"}

    result = run_from_text(view, out.raw_text)
    assert result.parse_result.schema_valid
    # gate admitted all three edges
    assert all(d["admitted"] for d in result.gate_decisions)

    statuses = result.final_belief_statuses
    assert statuses["b_old"] == "SUPERSEDED"
    assert statuses["b_new"] == "AUTHORIZED"
    assert statuses["b_dep"] == "AUTHORIZED"

    # audit trace carries gate decisions + an auditable supersede defeat path
    assert result.gate_decisions
    assert {"edge_type", "target_id", "admitted"} <= set(result.gate_decisions[0])
    supersede_path = next(dp for dp in result.defeat_paths if dp["belief_id"] == "b_old")
    assert supersede_path["path_type"] == "DIRECT_SUPERSEDE"
    assert supersede_path["replacement_belief_id"] == "b_new"
    # defeat path points back at the admitted evidence edge (grounding pointer)
    assert supersede_path["evidence_edge_ids"]
    assert set(result.audit_trace) >= {"edge_proposals", "defeat_paths", "fine_grained_statuses"}


def test_authorized_and_excluded_ids():
    ep = get_smoke_episode()
    result = run_from_text(ep.build_view(), ScriptedProposer(list(ep.gold_actions)).propose(ep.build_view()).raw_text)
    assert "b_old" in result.excluded_belief_ids
    assert "b_dep" in result.authorized_belief_ids


def test_exporters_build_and_validate():
    assert len(build_graph_rows()) >= 1
    assert len(build_revision_rows()) >= 1
    rollouts = build_rollout_rows()
    assert len(rollouts) >= 1
    pairs = build_preference_pairs(rollouts)
    assert len(pairs) >= 1
    # chosen reward strictly greater than rejected in every pair
    assert all(p["chosen_reward"] > p["rejected_reward"] for p in pairs)


def test_eval_harness_oracle_scores_perfect():
    episodes = get_smoke_episode  # noqa: F841 - imported for clarity
    from retrace_learn.data.build_synthetic_raw_dialogue import build_synthetic_episodes

    graph_examples = [ep.to_graph_extraction_example() for ep in build_synthetic_episodes()]
    report = eval_graph(RuleBasedGraphExtractor(), graph_examples)
    assert report["valid_json"] == 1.0
    assert report["evidence_node_f1"] == 1.0
    assert report["belief_node_f1"] == 1.0

    rev_examples = [ep.to_typed_revision_example() for ep in build_synthetic_episodes()]
    pol = eval_policy(rev_examples, proposer_factory=lambda ex: ScriptedProposer(ex.gold_action_objects()))
    assert pol["final_status_accuracy_after_DPA"] == 1.0
    assert pol["exact_action_match"] == 1.0

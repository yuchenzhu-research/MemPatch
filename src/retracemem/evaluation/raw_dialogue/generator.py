"""Deterministic synthetic raw-dialogue generator.

Produces structured subagent dialogues, corresponding graph targets,
and revision actions based on a random seed, without external API calls.
"""
from __future__ import annotations

import random
from typing import Any

from retracemem.authorization import authorize, EvidenceProposalBatch
from retracemem.schemas import EvidenceEdge, EvidenceEdgeType
from retracemem.methods.contracts import SharedCandidateView


def _run_dpa_to_get_statuses(
    graph: dict[str, Any],
    new_evidence_id: str,
    gold_actions: list[dict[str, Any]],
) -> dict[str, str]:
    """Helper to execute the deterministic authorize kernel to get correct final statuses."""
    from retracemem.schemas import EvidenceNode, BeliefNode, ConditionNode, DependencyEdge

    evidence_context = []
    new_evidence_node = None
    for ev in graph["evidence_nodes"]:
        node = EvidenceNode(
            evidence_id=ev["evidence_id"],
            session_id=ev["session_id"],
            timestamp=ev.get("timestamp"),
            text=ev["text"],
            source_dataset=ev.get("source_dataset", "synth"),
            source_pointer=ev.get("source_pointer", "p"),
        )
        evidence_context.append(node)
        if ev["evidence_id"] == new_evidence_id:
            new_evidence_node = node

    if new_evidence_node is None:
        new_evidence_node = EvidenceNode(
            evidence_id=new_evidence_id,
            session_id="sess_synth",
            timestamp="2026-06-01T00:00:00Z",
            text="New evidence trigger.",
            source_dataset="synth",
            source_pointer="p",
        )
        evidence_context.append(new_evidence_node)

    candidate_beliefs = []
    for b in graph["belief_nodes"]:
        candidate_beliefs.append(
            BeliefNode(
                belief_id=b["belief_id"],
                proposition=b["proposition"],
                source_evidence_ids=tuple(b.get("source_evidence_ids", ())),
            )
        )

    candidate_replacement_beliefs = []
    for b in graph["candidate_replacement_beliefs"]:
        candidate_replacement_beliefs.append(
            BeliefNode(
                belief_id=b["belief_id"],
                proposition=b["proposition"],
                source_evidence_ids=tuple(b.get("source_evidence_ids", ())),
            )
        )

    conditions_by_belief_dict: dict[str, list[ConditionNode]] = {}
    cond_by_id = {}
    for c in graph["condition_nodes"]:
        cond_by_id[c["condition_id"]] = ConditionNode(
            condition_id=c["condition_id"],
            scope_id=c.get("scope_id", "global"),
            text=c["text"],
        )

    deps_by_belief_dict: dict[str, list[DependencyEdge]] = {}
    for edge in graph["dependency_edges"]:
        bid = edge["belief_id"]
        cid = edge["condition_id"]
        dep = DependencyEdge(
            edge_id=edge["edge_id"],
            belief_id=bid,
            condition_id=cid,
            inducer=edge.get("inducer", "synth"),
        )
        deps_by_belief_dict.setdefault(bid, []).append(dep)

        cond = cond_by_id.get(cid)
        if cond:
            conditions_by_belief_dict.setdefault(bid, []).append(cond)

    conditions_by_belief_tuple = tuple(
        (bid, tuple(conds)) for bid, conds in conditions_by_belief_dict.items()
    )
    deps_by_belief_tuple = tuple(
        (bid, tuple(deps)) for bid, deps in deps_by_belief_dict.items()
    )

    view = SharedCandidateView(
        instance_id="inst_synth",
        query_id="query_synth",
        query="what is the status?",
        new_evidence=new_evidence_node,
        evidence_context=tuple(evidence_context),
        candidate_beliefs=tuple(candidate_beliefs),
        candidate_replacement_beliefs=tuple(candidate_replacement_beliefs),
        candidate_conditions_by_belief=conditions_by_belief_tuple,
        dependency_edges_by_belief=deps_by_belief_tuple,
    )

    edges = []
    for idx, a in enumerate(gold_actions):
        if a["action_type"] == "NO_REVISION":
            continue
        kind = "belief" if a.get("target_belief_id") else "condition"
        tid = a.get("target_belief_id") or a.get("target_condition_id")
        edges.append(
            EvidenceEdge(
                edge_id=f"edge_synth_{idx}",
                edge_type=EvidenceEdgeType(a["action_type"]),
                evidence_id=a["evidence_ids"][0],
                target_kind=kind,
                target_id=tid,
                verifier="synth_verifier",
                replacement_belief_id=a.get("replacement_belief_id"),
            )
        )

    proposal = EvidenceProposalBatch(edges=tuple(edges))
    res = authorize(view, (proposal,))
    return res.trace["fine_grained_statuses"]


class SyntheticDialogueGenerator:
    """Deterministic synthetic raw-dialogue generator."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def generate_episode(self, episode_id: str) -> dict[str, Any]:
        """Generate a random but logically sound raw dialogue episode."""
        templates = ["infra_migration", "finance_audit", "sensor_alarm"]
        choice = self.rng.choice(templates)

        if choice == "infra_migration":
            db_name = self.rng.choice(["MySQL", "Postgres", "Redis", "MongoDB"])
            v_old = self.rng.choice(["5.7", "11.0", "6.2", "4.4"])
            v_new = self.rng.choice(["8.0", "15.0", "7.0", "6.0"])
            user_flag = self.rng.choice(["BillingFlag", "BetaTrial", "AIWorkspace", "FeatureX"])

            raw_dialogue = (
                f"Agent A (historian): @EVIDENCE ev1 [t=2024-01-01]: Legacy DB runs on {db_name} {v_old} in staging.\n"
                f"Agent A (historian): @BELIEF b_old <- ev1: Staging environment runs on {db_name} {v_old}.\n"
                f"Agent A (historian): @BELIEF b_dep <- ev1: Active features are accessible to beta testers.\n"
                f"Agent A (historian): @CONDITION c1 @{user_flag}: Feature flag {user_flag} must be enabled.\n"
                f"Agent A (historian): @REQUIRES b_dep -> c1\n"
                f"Agent B (engineer): @EVIDENCE ev2 [t=2024-02-01]: Upgrade successfully completed to {db_name} {v_new}.\n"
                f"Agent B (engineer): @REPLACEMENT b_new <- ev2: Staging environment runs on {db_name} {v_new}.\n"
                f"Agent C (sentry): @EVIDENCE ev3 [t=2024-03-01]: Outage: {user_flag} flag turned off.\n"
                f"Agent D (oncall): @EVIDENCE ev4 [t=2024-04-01]: Resolution: {user_flag} flag re-enabled."
            )

            evidence_nodes = [
                {"evidence_id": "ev1", "session_id": "sess_1", "text": f"Legacy DB runs on {db_name} {v_old} in staging."},
                {"evidence_id": "ev2", "session_id": "sess_1", "text": f"Upgrade successfully completed to {db_name} {v_new}."},
                {"evidence_id": "ev3", "session_id": "sess_1", "text": f"Outage: {user_flag} flag turned off."},
                {"evidence_id": "ev4", "session_id": "sess_1", "text": f"Resolution: {user_flag} flag re-enabled."},
            ]

            belief_nodes = [
                {"belief_id": "b_old", "proposition": f"Staging environment runs on {db_name} {v_old}.", "source_evidence_ids": ["ev1"]},
                {"belief_id": "b_dep", "proposition": "Active features are accessible to beta testers.", "source_evidence_ids": ["ev1"]},
            ]

            condition_nodes = [
                {"condition_id": "c1", "scope_id": user_flag, "text": f"Feature flag {user_flag} must be enabled."}
            ]

            candidate_replacement_beliefs = [
                {"belief_id": "b_new", "proposition": f"Staging environment runs on {db_name} {v_new}.", "source_evidence_ids": ["ev2"]}
            ]

            dependency_edges = [
                {"edge_id": "dep_1", "belief_id": "b_dep", "condition_id": "c1", "inducer": "location_service", "edge_type": "REQUIRES"}
            ]

            gold_actions = [
                {
                    "action_type": "SUPERSEDES",
                    "target_belief_id": "b_old",
                    "replacement_belief_id": "b_new",
                    "evidence_ids": ["ev2"],
                    "rationale": "migration complete",
                },
                {
                    "action_type": "BLOCKS",
                    "target_condition_id": "c1",
                    "evidence_ids": ["ev3"],
                    "rationale": "flag turned off",
                },
                {
                    "action_type": "RELEASES",
                    "target_condition_id": "c1",
                    "evidence_ids": ["ev4"],
                    "rationale": "flag re-enabled",
                },
            ]

            new_evidence_id = "ev4"

        elif choice == "finance_audit":
            quarter = self.rng.choice(["Q1", "Q2", "Q3", "Q4"])
            rev_val = self.rng.choice(["10M", "15M", "20M", "5M"])
            auditor_name = self.rng.choice(["KPMG", "PwC", "EY", "Deloitte"])

            raw_dialogue = (
                f"Agent A (finance): @EVIDENCE ev1 [t=2024-01-01]: Finance ledger indicates {quarter} revenue is {rev_val} USD.\n"
                f"Agent A (finance): @BELIEF b1 <- ev1: {quarter} revenue was {rev_val} USD.\n"
                f"Agent B (auditor): @EVIDENCE ev2 [t=2024-02-01]: Official audit by {auditor_name} confirms {quarter} revenue of {rev_val} USD."
            )

            evidence_nodes = [
                {"evidence_id": "ev1", "session_id": "sess_2", "text": f"Finance ledger indicates {quarter} revenue is {rev_val} USD."},
                {"evidence_id": "ev2", "session_id": "sess_2", "text": f"Official audit by {auditor_name} confirms {quarter} revenue of {rev_val} USD."},
            ]

            belief_nodes = [
                {"belief_id": "b1", "proposition": f"{quarter} revenue was {rev_val} USD.", "source_evidence_ids": ["ev1"]}
            ]

            condition_nodes = []
            candidate_replacement_beliefs = []
            dependency_edges = []

            gold_actions = [
                {
                    "action_type": "REAFFIRMS",
                    "target_belief_id": "b1",
                    "evidence_ids": ["ev2"],
                    "rationale": "independent audit verification",
                }
            ]

            new_evidence_id = "ev2"

        else:
            sensor_id = self.rng.choice(["SN-900", "SN-402", "SN-109"])
            limit_val = self.rng.choice(["50 PSI", "100 PSI", "200 PSI"])
            conflict_val = self.rng.choice(["75 PSI", "120 PSI", "250 PSI"])

            raw_dialogue = (
                f"Agent A (sensor_hub): @EVIDENCE ev1 [t=2024-01-01]: Sensor {sensor_id} reports stable pressure at {limit_val}.\n"
                f"Agent A (sensor_hub): @BELIEF b1 <- ev1: System pressure is {limit_val}.\n"
                f"Agent B (sentry): @EVIDENCE ev2 [t=2024-02-01]: Sensor {sensor_id} reports anomalous reading of {conflict_val} with calibration error."
            )

            evidence_nodes = [
                {"evidence_id": "ev1", "session_id": "sess_3", "text": f"Sensor {sensor_id} reports stable pressure at {limit_val}."},
                {"evidence_id": "ev2", "session_id": "sess_3", "text": f"Sensor {sensor_id} reports anomalous reading of {conflict_val} with calibration error."},
            ]

            belief_nodes = [
                {"belief_id": "b1", "proposition": f"System pressure is {limit_val}.", "source_evidence_ids": ["ev1"]}
            ]

            condition_nodes = []
            candidate_replacement_beliefs = []
            dependency_edges = []

            gold_actions = [
                {
                    "action_type": "UNCERTAIN",
                    "target_belief_id": "b1",
                    "evidence_ids": ["ev2"],
                    "rationale": "conflicting uncalibrated reading",
                }
            ]

            new_evidence_id = "ev2"

        gold_graph = {
            "evidence_nodes": evidence_nodes,
            "belief_nodes": belief_nodes,
            "condition_nodes": condition_nodes,
            "candidate_replacement_beliefs": candidate_replacement_beliefs,
            "dependency_edges": dependency_edges,
        }

        gold_final_statuses = _run_dpa_to_get_statuses(gold_graph, new_evidence_id, gold_actions)

        return {
            "example_id": episode_id,
            "raw_dialogue": raw_dialogue,
            "subagent_roles": ["historian", "engineer", "sentry", "oncall", "finance", "auditor", "sensor_hub"],
            "gold_graph": gold_graph,
            "new_evidence_id": new_evidence_id,
            "gold_actions": gold_actions,
            "gold_final_statuses": gold_final_statuses,
        }

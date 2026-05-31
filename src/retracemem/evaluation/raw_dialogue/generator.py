"""Deterministic synthetic raw-dialogue generator.

Produces structured subagent dialogues, corresponding graph targets,
and revision actions based on a random seed, supporting 13 case families.
"""
from __future__ import annotations

import random
from typing import Any

from retracemem.authorization import authorize, EvidenceProposalBatch
from retracemem.schemas import EvidenceEdge, EvidenceEdgeType, EvidenceNode, BeliefNode, ConditionNode, DependencyEdge
from retracemem.methods.contracts import SharedCandidateView


def _run_dpa_to_get_statuses(
    graph: dict[str, Any],
    new_evidence_id: str,
    gold_actions: list[dict[str, Any]],
) -> dict[str, str]:
    """Helper to execute the deterministic authorize kernel to get correct final statuses."""
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
    """Deterministic synthetic raw-dialogue generator supporting 13 case families."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.case_families = [
            "supersedes_basic",
            "blocks_basic",
            "releases_basic",
            "uncertain_basic",
            "reaffirms_basic",
            "no_revision_basic",
            "supersedes_blocks_multi",
            "blocks_then_releases_temporal",
            "ambiguous_target",
            "irrelevant_reaffirmation",
            "duplicate_evidence",
            "scope_leakage",
            "stale_propagation_trap"
        ]

    def generate_episode(self, episode_id: str, case_family: str | None = None) -> dict[str, Any]:
        """Generate a random but logically sound raw dialogue episode."""
        if case_family is None:
            case_family = self.rng.choice(self.case_families)

        if case_family == "supersedes_basic":
            raw_dialogue = (
                "Agent A: @EVIDENCE ev1 [t=2024-01-01]: Legacy DB runs on MySQL 5.7.\n"
                "Agent A: @BELIEF b_old <- ev1: Staging runs on MySQL 5.7.\n"
                "Agent B: @EVIDENCE ev2 [t=2024-02-01]: Upgrade successfully completed to MySQL 8.0.\n"
                "Agent B: @REPLACEMENT b_new <- ev2: Staging runs on MySQL 8.0."
            )
            evidence_nodes = [
                {"evidence_id": "ev1", "session_id": "sess_1", "text": "Legacy DB runs on MySQL 5.7."},
                {"evidence_id": "ev2", "session_id": "sess_1", "text": "Upgrade successfully completed to MySQL 8.0."},
            ]
            belief_nodes = [{"belief_id": "b_old", "proposition": "Staging runs on MySQL 5.7.", "source_evidence_ids": ["ev1"]}]
            condition_nodes = []
            candidate_replacement_beliefs = [{"belief_id": "b_new", "proposition": "Staging runs on MySQL 8.0.", "source_evidence_ids": ["ev2"]}]
            dependency_edges = []
            gold_actions = [{
                "action_type": "SUPERSEDES",
                "target_belief_id": "b_old",
                "replacement_belief_id": "b_new",
                "evidence_ids": ["ev2"],
                "rationale": "migration replacement"
            }]
            new_evidence_id = "ev2"

        elif case_family == "blocks_basic":
            raw_dialogue = (
                "Agent A: @EVIDENCE ev1 [t=2024-01-01]: App relies on API access.\n"
                "Agent A: @BELIEF b1 <- ev1: Application is currently running.\n"
                "Agent A: @CONDITION c1 @api_link: API link is available.\n"
                "Agent A: @REQUIRES b1 -> c1\n"
                "Agent B: @EVIDENCE ev2 [t=2024-02-01]: API server key revoked by security."
            )
            evidence_nodes = [
                {"evidence_id": "ev1", "session_id": "sess_2", "text": "App relies on API access."},
                {"evidence_id": "ev2", "session_id": "sess_2", "text": "API server key revoked by security."},
            ]
            belief_nodes = [{"belief_id": "b1", "proposition": "Application is currently running.", "source_evidence_ids": ["ev1"]}]
            condition_nodes = [{"condition_id": "c1", "scope_id": "api_link", "text": "API link is available."}]
            candidate_replacement_beliefs = []
            dependency_edges = [{"edge_id": "dep_1", "belief_id": "b1", "condition_id": "c1", "inducer": "auth"}]
            gold_actions = [{
                "action_type": "BLOCKS",
                "target_condition_id": "c1",
                "evidence_ids": ["ev2"],
                "rationale": "key revoked"
            }]
            new_evidence_id = "ev2"

        elif case_family == "releases_basic":
            raw_dialogue = (
                "Agent A: @EVIDENCE ev1 [t=2024-01-01]: Service requires storage.\n"
                "Agent A: @BELIEF b1 <- ev1: Storage mount is functional.\n"
                "Agent A: @CONDITION c1 @mount_check: Disk quota is under limit.\n"
                "Agent A: @REQUIRES b1 -> c1\n"
                "Agent B: @EVIDENCE ev2 [t=2024-02-01]: SRE cleared logs and storage space restored."
            )
            evidence_nodes = [
                {"evidence_id": "ev1", "session_id": "sess_3", "text": "Service requires storage."},
                {"evidence_id": "ev2", "session_id": "sess_3", "text": "SRE cleared logs and storage space restored."},
            ]
            belief_nodes = [{"belief_id": "b1", "proposition": "Storage mount is functional.", "source_evidence_ids": ["ev1"]}]
            condition_nodes = [{"condition_id": "c1", "scope_id": "mount_check", "text": "Disk quota is under limit."}]
            candidate_replacement_beliefs = []
            dependency_edges = [{"edge_id": "dep_1", "belief_id": "b1", "condition_id": "c1", "inducer": "sre"}]
            gold_actions = [{
                "action_type": "RELEASES",
                "target_condition_id": "c1",
                "evidence_ids": ["ev2"],
                "rationale": "SRE released disk quota block"
            }]
            new_evidence_id = "ev2"

        elif case_family == "uncertain_basic":
            raw_dialogue = (
                "Agent A: @EVIDENCE ev1 [t=2024-01-01]: Server is healthy.\n"
                "Agent A: @BELIEF b1 <- ev1: Web application is online.\n"
                "Agent B: @EVIDENCE ev2 [t=2024-02-01]: Intermittent connection timeouts observed."
            )
            evidence_nodes = [
                {"evidence_id": "ev1", "session_id": "sess_4", "text": "Server is healthy."},
                {"evidence_id": "ev2", "session_id": "sess_4", "text": "Intermittent connection timeouts observed."},
            ]
            belief_nodes = [{"belief_id": "b1", "proposition": "Web application is online.", "source_evidence_ids": ["ev1"]}]
            condition_nodes = []
            candidate_replacement_beliefs = []
            dependency_edges = []
            gold_actions = [{
                "action_type": "UNCERTAIN",
                "target_belief_id": "b1",
                "evidence_ids": ["ev2"],
                "rationale": "timeouts observed"
            }]
            new_evidence_id = "ev2"

        elif case_family == "reaffirms_basic":
            raw_dialogue = (
                "Agent A: @EVIDENCE ev1 [t=2024-01-01]: Server located in Seattle.\n"
                "Agent A: @BELIEF b1 <- ev1: Server resides in Seattle dataclass.\n"
                "Agent B: @EVIDENCE ev2 [t=2024-02-01]: GeoIP check confirms location is Seattle."
            )
            evidence_nodes = [
                {"evidence_id": "ev1", "session_id": "sess_5", "text": "Server located in Seattle."},
                {"evidence_id": "ev2", "session_id": "sess_5", "text": "GeoIP check confirms location is Seattle."},
            ]
            belief_nodes = [{"belief_id": "b1", "proposition": "Server resides in Seattle dataclass.", "source_evidence_ids": ["ev1"]}]
            condition_nodes = []
            candidate_replacement_beliefs = []
            dependency_edges = []
            gold_actions = [{
                "action_type": "REAFFIRMS",
                "target_belief_id": "b1",
                "evidence_ids": ["ev2"],
                "rationale": "location reaffirmed"
            }]
            new_evidence_id = "ev2"

        elif case_family == "no_revision_basic":
            raw_dialogue = (
                "Agent A: @EVIDENCE ev1 [t=2024-01-01]: Server is running.\n"
                "Agent A: @BELIEF b1 <- ev1: Web application is online.\n"
                "Agent B: @EVIDENCE ev2 [t=2024-02-01]: Cafeteria serves lunch until 2pm."
            )
            evidence_nodes = [
                {"evidence_id": "ev1", "session_id": "sess_6", "text": "Server is running."},
                {"evidence_id": "ev2", "session_id": "sess_6", "text": "Cafeteria serves lunch until 2pm."},
            ]
            belief_nodes = [{"belief_id": "b1", "proposition": "Web application is online.", "source_evidence_ids": ["ev1"]}]
            condition_nodes = []
            candidate_replacement_beliefs = []
            dependency_edges = []
            gold_actions = [{
                "action_type": "NO_REVISION",
                "evidence_ids": ["ev2"],
                "rationale": "irrelevant lunchtime info"
            }]
            new_evidence_id = "ev2"

        elif case_family == "supersedes_blocks_multi":
            raw_dialogue = (
                "Agent A: @EVIDENCE ev1 [t=2024-01-01]: Dev database is MySQL 5.7.\n"
                "Agent A: @BELIEF b_old <- ev1: Dev database runs MySQL 5.7.\n"
                "Agent A: @BELIEF b_dep <- ev1: QA suite is fully functional.\n"
                "Agent A: @CONDITION c1 @qa_run: QA docker environment active.\n"
                "Agent A: @REQUIRES b_dep -> c1\n"
                "Agent B: @EVIDENCE ev2 [t=2024-02-01]: Database migrated to MySQL 8.0.\n"
                "Agent B: @REPLACEMENT b_new <- ev2: Dev database runs MySQL 8.0.\n"
                "Agent C: @EVIDENCE ev3 [t=2024-03-01]: QA docker host failed disk allocation."
            )
            evidence_nodes = [
                {"evidence_id": "ev1", "session_id": "sess_7", "text": "Dev database is MySQL 5.7."},
                {"evidence_id": "ev2", "session_id": "sess_7", "text": "Database migrated to MySQL 8.0."},
                {"evidence_id": "ev3", "session_id": "sess_7", "text": "QA docker host failed disk allocation."},
            ]
            belief_nodes = [
                {"belief_id": "b_old", "proposition": "Dev database runs MySQL 5.7.", "source_evidence_ids": ["ev1"]},
                {"belief_id": "b_dep", "proposition": "QA suite is fully functional.", "source_evidence_ids": ["ev1"]},
            ]
            condition_nodes = [{"condition_id": "c1", "scope_id": "qa_run", "text": "QA docker environment active."}]
            candidate_replacement_beliefs = [{"belief_id": "b_new", "proposition": "Dev database runs MySQL 8.0.", "source_evidence_ids": ["ev2"]}]
            dependency_edges = [{"edge_id": "dep_1", "belief_id": "b_dep", "condition_id": "c1", "inducer": "qa"}]
            gold_actions = [
                {
                    "action_type": "SUPERSEDES",
                    "target_belief_id": "b_old",
                    "replacement_belief_id": "b_new",
                    "evidence_ids": ["ev2"],
                    "rationale": "upgrade mysql"
                },
                {
                    "action_type": "BLOCKS",
                    "target_condition_id": "c1",
                    "evidence_ids": ["ev3"],
                    "rationale": "disk allocation failure"
                }
            ]
            new_evidence_id = "ev3"

        elif case_family == "blocks_then_releases_temporal":
            raw_dialogue = (
                "Agent A: @EVIDENCE ev1 [t=2024-01-01]: Server active.\n"
                "Agent A: @BELIEF b1 <- ev1: Web application online.\n"
                "Agent A: @CONDITION c1 @net: Network port is open.\n"
                "Agent A: @REQUIRES b1 -> c1\n"
                "Agent B: @EVIDENCE ev2 [t=2024-02-01]: Security closed port due to vulnerability.\n"
                "Agent C: @EVIDENCE ev3 [t=2024-03-01]: Firewall rules updated; port reopened."
            )
            evidence_nodes = [
                {"evidence_id": "ev1", "session_id": "sess_8", "text": "Server active."},
                {"evidence_id": "ev2", "session_id": "sess_8", "text": "Security closed port due to vulnerability."},
                {"evidence_id": "ev3", "session_id": "sess_8", "text": "Firewall rules updated; port reopened."},
            ]
            belief_nodes = [{"belief_id": "b1", "proposition": "Web application online.", "source_evidence_ids": ["ev1"]}]
            condition_nodes = [{"condition_id": "c1", "scope_id": "net", "text": "Network port is open."}]
            candidate_replacement_beliefs = []
            dependency_edges = [{"edge_id": "dep_1", "belief_id": "b1", "condition_id": "c1", "inducer": "firewall"}]
            gold_actions = [
                {
                    "action_type": "BLOCKS",
                    "target_condition_id": "c1",
                    "evidence_ids": ["ev2"],
                    "rationale": "vulnerability mitigation"
                },
                {
                    "action_type": "RELEASES",
                    "target_condition_id": "c1",
                    "evidence_ids": ["ev3"],
                    "rationale": "firewall rules updated"
                }
            ]
            new_evidence_id = "ev3"

        elif case_family == "ambiguous_target":
            raw_dialogue = (
                "Agent A: @EVIDENCE ev1 [t=2024-01-01]: Office location is Seattle.\n"
                "Agent A: @BELIEF b_old <- ev1: Office is located in Seattle.\n"
                "Agent B: @EVIDENCE ev2 [t=2024-02-01]: Relocated office to Seattle branch."
            )
            evidence_nodes = [
                {"evidence_id": "ev1", "session_id": "sess_9", "text": "Office location is Seattle."},
                {"evidence_id": "ev2", "session_id": "sess_9", "text": "Relocated office to Seattle branch."},
            ]
            belief_nodes = [{"belief_id": "b_old", "proposition": "Office is located in Seattle.", "source_evidence_ids": ["ev1"]}]
            condition_nodes = []
            candidate_replacement_beliefs = []
            dependency_edges = []
            gold_actions = [{
                "action_type": "REAFFIRMS",
                "target_belief_id": "b_old",
                "evidence_ids": ["ev2"],
                "rationale": "office branch reaffirms Seattle"
            }]
            new_evidence_id = "ev2"

        elif case_family == "irrelevant_reaffirmation":
            raw_dialogue = (
                "Agent A: @EVIDENCE ev1 [t=2024-01-01]: Server resides in Seattle dataclass.\n"
                "Agent A: @BELIEF b1 <- ev1: Server Seattle office location.\n"
                "Agent B: @EVIDENCE ev2 [t=2024-02-01]: Datacenter power backup generator tested OK."
            )
            evidence_nodes = [
                {"evidence_id": "ev1", "session_id": "sess_10", "text": "Server resides in Seattle dataclass."},
                {"evidence_id": "ev2", "session_id": "sess_10", "text": "Datacenter power backup generator tested OK."},
            ]
            belief_nodes = [{"belief_id": "b1", "proposition": "Server Seattle office location.", "source_evidence_ids": ["ev1"]}]
            condition_nodes = []
            candidate_replacement_beliefs = []
            dependency_edges = []
            gold_actions = [{
                "action_type": "NO_REVISION",
                "evidence_ids": ["ev2"],
                "rationale": "power backup test irrelevant to location belief"
            }]
            new_evidence_id = "ev2"

        elif case_family == "duplicate_evidence":
            raw_dialogue = (
                "Agent A: @EVIDENCE ev1 [t=2024-01-01]: App database is Postgres.\n"
                "Agent A: @BELIEF b1 <- ev1: App database runs Postgres.\n"
                "Agent B: @EVIDENCE ev2 [t=2024-02-01]: System uses PostgreSQL database.\n"
                "Agent C: @EVIDENCE ev3 [t=2024-03-01]: Confirmation: PostgreSQL database is in place."
            )
            evidence_nodes = [
                {"evidence_id": "ev1", "session_id": "sess_11", "text": "App database is Postgres."},
                {"evidence_id": "ev2", "session_id": "sess_11", "text": "System uses PostgreSQL database."},
                {"evidence_id": "ev3", "session_id": "sess_11", "text": "Confirmation: PostgreSQL database is in place."},
            ]
            belief_nodes = [{"belief_id": "b1", "proposition": "App database runs Postgres.", "source_evidence_ids": ["ev1"]}]
            condition_nodes = []
            candidate_replacement_beliefs = []
            dependency_edges = []
            gold_actions = [
                {
                    "action_type": "REAFFIRMS",
                    "target_belief_id": "b1",
                    "evidence_ids": ["ev2"],
                    "rationale": "first reaffirmation"
                },
                {
                    "action_type": "REAFFIRMS",
                    "target_belief_id": "b1",
                    "evidence_ids": ["ev3"],
                    "rationale": "duplicate confirmation"
                }
            ]
            new_evidence_id = "ev3"

        elif case_family == "scope_leakage":
            raw_dialogue = (
                "Agent A: @EVIDENCE ev1 [t=2024-01-01]: User is beta customer.\n"
                "Agent A: @BELIEF b1 <- ev1: User is eligible for AI workspace.\n"
                "Agent A: @CONDITION c1 @beta: Beta scope active.\n"
                "Agent A: @REQUIRES b1 -> c1\n"
                "Agent B: @EVIDENCE ev2 [t=2024-02-01]: User downgraded account to free tier."
            )
            evidence_nodes = [
                {"evidence_id": "ev1", "session_id": "sess_12", "text": "User is beta customer."},
                {"evidence_id": "ev2", "session_id": "sess_12", "text": "User downgraded account to free tier."},
            ]
            belief_nodes = [{"belief_id": "b1", "proposition": "User is eligible for AI workspace.", "source_evidence_ids": ["ev1"]}]
            condition_nodes = [{"condition_id": "c1", "scope_id": "beta", "text": "Beta scope active."}]
            candidate_replacement_beliefs = []
            dependency_edges = [{"edge_id": "dep_1", "belief_id": "b1", "condition_id": "c1", "inducer": "beta"}]
            gold_actions = [{
                "action_type": "BLOCKS",
                "target_condition_id": "c1",
                "evidence_ids": ["ev2"],
                "rationale": "downgrade blocks beta scope"
            }]
            new_evidence_id = "ev2"

        else:  # stale_propagation_trap
            raw_dialogue = (
                "Agent A: @EVIDENCE ev1 [t=2024-01-01]: Core database runs MySQL 5.7.\n"
                "Agent A: @BELIEF b_old <- ev1: Core database runs MySQL 5.7.\n"
                "Agent A: @BELIEF b_dep <- ev1: System dashboard is active.\n"
                "Agent A: @CONDITION c1 @db_status: Database is healthy and reachable.\n"
                "Agent A: @REQUIRES b_dep -> c1\n"
                "Agent B: @EVIDENCE ev2 [t=2024-02-01]: Outage: Core DB crashed with error."
            )
            evidence_nodes = [
                {"evidence_id": "ev1", "session_id": "sess_13", "text": "Core database runs MySQL 5.7."},
                {"evidence_id": "ev2", "session_id": "sess_13", "text": "Outage: Core DB crashed with error."},
            ]
            belief_nodes = [
                {"belief_id": "b_old", "proposition": "Core database runs MySQL 5.7.", "source_evidence_ids": ["ev1"]},
                {"belief_id": "b_dep", "proposition": "System dashboard is active.", "source_evidence_ids": ["ev1"]},
            ]
            condition_nodes = [{"condition_id": "c1", "scope_id": "db_status", "text": "Database is healthy and reachable."}]
            candidate_replacement_beliefs = []
            dependency_edges = [{"edge_id": "dep_1", "belief_id": "b_dep", "condition_id": "c1", "inducer": "db"}]
            gold_actions = [{
                "action_type": "BLOCKS",
                "target_condition_id": "c1",
                "evidence_ids": ["ev2"],
                "rationale": "outage blocks db_status condition"
            }]
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
            "case_family": case_family,
            "raw_dialogue": raw_dialogue,
            "subagent_roles": ["historian", "engineer", "sentry", "oncall", "finance", "auditor", "sensor_hub"],
            "gold_graph": gold_graph,
            "new_evidence_id": new_evidence_id,
            "gold_actions": gold_actions,
            "gold_final_statuses": gold_final_statuses,
        }

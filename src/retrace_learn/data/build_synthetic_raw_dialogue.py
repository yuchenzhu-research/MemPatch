"""Smoke/sanity synthetic dialogues + gold labels for ReTrace-Learn.

This is a **smoke/sanity generator only** \u2014 NOT the large-scale ReTrace-Learn
training generator. It exists to exercise the full pipeline (Graph Builder ->
Proposal Policy -> DPA commit) end to end and to give the schema/reward tests one
worked example per canonical action. The future large-scale training corpus is
out of scope here and should be generated under ``data/retrace_learn/v1_0/``.

Each :class:`SyntheticEpisode` carries a hand-authored dialogue (written in the
ReTrace-Learn markup so the rule-based extractor recovers the graph exactly), the
gold memory graph, the triggering ``new_evidence``, and gold typed actions. Gold
final statuses are NOT hand-written: they are computed by running the gold
actions through the real DPA runtime, so labels are kernel-consistent by
construction.

These episodes cover all six canonical actions:
  smoke         -> SUPERSEDES + BLOCKS + RELEASES
  reaffirm      -> REAFFIRMS
  uncertain     -> UNCERTAIN
  no_revision   -> NO_REVISION
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retrace_learn.schemas import RevisionAction, TypedRevisionExample, GraphExtractionExample
from retrace_learn.runtime.graph_extractor import RuleBasedGraphExtractor
from retrace_learn.runtime.views import build_view
from retrace_learn.runtime.dpa_runtime import run_actions


@dataclass(frozen=True)
class SyntheticEpisode:
    example_id: str
    domain: str
    subagent_roles: list[str]
    raw_dialogue: str
    new_evidence_id: str
    query: str
    gold_actions: list[RevisionAction]
    instance_id: str = "instance_0"
    query_id: str = "query_0"
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- derived structure -------------------------------------------------
    def graph(self) -> dict[str, Any]:
        return RuleBasedGraphExtractor().extract(self.raw_dialogue, self.subagent_roles)

    def _view_fields(self) -> dict[str, Any]:
        g = self.graph()
        candidate_beliefs = list(g["belief_nodes"])
        conditions_by_belief: dict[str, list[dict[str, Any]]] = {}
        deps_by_belief: dict[str, list[dict[str, Any]]] = {}
        cond_by_id = {c["condition_id"]: c for c in g["condition_nodes"]}
        for edge in g["dependency_edges"]:
            bid = edge["belief_id"]
            deps_by_belief.setdefault(bid, []).append(edge)
            cond = cond_by_id.get(edge["condition_id"])
            if cond is not None:
                conditions_by_belief.setdefault(bid, []).append(cond)
        return {
            "evidence_context": g["evidence_nodes"],
            "candidate_beliefs": candidate_beliefs,
            "candidate_replacement_beliefs": list(g["candidate_replacement_beliefs"]),
            "candidate_conditions_by_belief": conditions_by_belief,
            "dependency_edges_by_belief": deps_by_belief,
        }

    def build_view(self):
        fields = self._view_fields()
        return build_view(
            instance_id=self.instance_id,
            query_id=self.query_id,
            query=self.query,
            new_evidence_id=self.new_evidence_id,
            **fields,
        )

    def gold_final_statuses(self) -> dict[str, str]:
        view = self.build_view()
        result = run_actions(view, list(self.gold_actions))
        return result.final_belief_statuses

    # -- export helpers ----------------------------------------------------
    def to_graph_extraction_example(self) -> GraphExtractionExample:
        return GraphExtractionExample(
            example_id=f"{self.example_id}_graph",
            raw_dialogue=self.raw_dialogue,
            subagent_roles=self.subagent_roles,
            output_graph=self.graph(),
            metadata={"domain": self.domain, "source": "synthetic_v1"},
        )

    def to_typed_revision_example(self) -> TypedRevisionExample:
        g = self.graph()
        fields = self._view_fields()
        return TypedRevisionExample(
            example_id=f"{self.example_id}_rev",
            current_graph=g,
            new_evidence=next(
                e for e in g["evidence_nodes"] if e["evidence_id"] == self.new_evidence_id
            ),
            candidate_beliefs=fields["candidate_beliefs"],
            candidate_replacement_beliefs=fields["candidate_replacement_beliefs"],
            candidate_conditions_by_belief=fields["candidate_conditions_by_belief"],
            dependency_edges_by_belief=fields["dependency_edges_by_belief"],
            gold_actions=[a.to_dict() for a in self.gold_actions],
            gold_final_statuses=self.gold_final_statuses(),
            metadata={"domain": self.domain, "source": "synthetic_v1"},
        )


# ---------------------------------------------------------------------------
# Episode definitions
# ---------------------------------------------------------------------------

_SMOKE_DIALOGUE = """\
Subagent A (historian): @EVIDENCE ev1 [t=2024-01-01]: Service runs on MySQL 5.7 in production.
Subagent A (historian): @BELIEF b_old <- ev1: The service database engine is MySQL 5.7.
Subagent A (historian): @BELIEF b_dep <- ev1: Feature X is available to end users.
Subagent A (historian): @CONDITION c1 @feature_x: Feature flag X is enabled.
Subagent A (historian): @REQUIRES b_dep -> c1
Subagent B (migrator): @EVIDENCE ev2 [t=2024-02-01]: Migration finished; service now runs MySQL 8.0.
Subagent B (migrator): @REPLACEMENT b_new <- ev2: The service database engine is MySQL 8.0.
Subagent C (oncall): @EVIDENCE ev3 [t=2024-03-01]: Incident: feature flag X disabled pending review.
Subagent D (reviewer): @EVIDENCE ev4 [t=2024-04-01]: Review complete; feature flag X re-enabled.
"""

_REAFFIRM_DIALOGUE = """\
Subagent A (analyst): @EVIDENCE ev1 [t=2024-01-01]: Q3 revenue reported as 12M USD.
Subagent A (analyst): @BELIEF b1 <- ev1: Q3 revenue was 12M USD.
Subagent B (auditor): @EVIDENCE ev2 [t=2024-02-01]: Independent audit confirms Q3 revenue of 12M USD.
"""

_UNCERTAIN_DIALOGUE = """\
Subagent A (sensor): @EVIDENCE ev1 [t=2024-01-01]: Sensor reports tank pressure is 30 PSI.
Subagent A (sensor): @BELIEF b1 <- ev1: Tank pressure is 30 PSI.
Subagent B (sensor): @EVIDENCE ev2 [t=2024-02-01]: A second sensor reports a conflicting 45 PSI with no calibration record.
"""

_NO_REVISION_DIALOGUE = """\
Subagent A (planner): @EVIDENCE ev1 [t=2024-01-01]: The office is located at 5th Avenue.
Subagent A (planner): @BELIEF b1 <- ev1: The office is on 5th Avenue.
Subagent B (intern): @EVIDENCE ev2 [t=2024-02-01]: The cafeteria serves coffee until 4pm.
"""


def build_synthetic_episodes() -> list[SyntheticEpisode]:
    return [
        SyntheticEpisode(
            example_id="ep_smoke",
            domain="infra",
            subagent_roles=["historian", "migrator", "oncall", "reviewer"],
            raw_dialogue=_SMOKE_DIALOGUE,
            new_evidence_id="ev4",
            query="What is the current authorized database engine and is Feature X usable?",
            gold_actions=[
                RevisionAction(
                    action_type="SUPERSEDES",
                    target_belief_id="b_old",
                    replacement_belief_id="b_new",
                    evidence_ids=("ev2",),
                    rationale="ev2 reports migration to MySQL 8.0, replacing b_old.",
                ),
                RevisionAction(
                    action_type="BLOCKS",
                    target_condition_id="c1",
                    evidence_ids=("ev3",),
                    rationale="ev3 disables feature flag X, blocking condition c1.",
                ),
                RevisionAction(
                    action_type="RELEASES",
                    target_condition_id="c1",
                    evidence_ids=("ev4",),
                    rationale="ev4 re-enables feature flag X, releasing condition c1.",
                ),
            ],
        ),
        SyntheticEpisode(
            example_id="ep_reaffirm",
            domain="finance",
            subagent_roles=["analyst", "auditor"],
            raw_dialogue=_REAFFIRM_DIALOGUE,
            new_evidence_id="ev2",
            query="What was Q3 revenue?",
            gold_actions=[
                RevisionAction(
                    action_type="REAFFIRMS",
                    target_belief_id="b1",
                    evidence_ids=("ev2",),
                    rationale="ev2 independently confirms b1.",
                )
            ],
        ),
        SyntheticEpisode(
            example_id="ep_uncertain",
            domain="ops",
            subagent_roles=["sensor", "sensor"],
            raw_dialogue=_UNCERTAIN_DIALOGUE,
            new_evidence_id="ev2",
            query="What is the current tank pressure?",
            gold_actions=[
                RevisionAction(
                    action_type="UNCERTAIN",
                    target_belief_id="b1",
                    evidence_ids=("ev2",),
                    rationale="ev2 conflicts with b1 and has no calibration record.",
                )
            ],
        ),
        SyntheticEpisode(
            example_id="ep_no_revision",
            domain="facilities",
            subagent_roles=["planner", "intern"],
            raw_dialogue=_NO_REVISION_DIALOGUE,
            new_evidence_id="ev2",
            query="Where is the office located?",
            gold_actions=[
                RevisionAction(
                    action_type="NO_REVISION",
                    evidence_ids=("ev2",),
                    rationale="ev2 is irrelevant to b1.",
                )
            ],
        ),
    ]


def get_smoke_episode() -> SyntheticEpisode:
    return build_synthetic_episodes()[0]

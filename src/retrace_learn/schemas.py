"""MemPatch scaffold training-data contracts.

These dataclasses define JSONL training schemas for the MemPatch pipeline
(scenario/event_trace -> revision view -> benchmark-compatible response ->
benchmark-grounded feedback). They align with the canonical runtime vocabulary
in ``retracemem``:

* typed action vocabulary  -> :data:`CANONICAL_ACTIONS`
  (``SUPERSEDES``/``BLOCKS``/``RELEASES``/``UNCERTAIN``/``REAFFIRMS``/``NO_REVISION``)
* DPA final status vocabulary -> :data:`FINAL_STATUSES`
  (``AUTHORIZED``/``SUPERSEDED``/``BLOCKED``/``UNRESOLVED``)
* candidate defeat-path types -> :data:`CANDIDATE_PATH_TYPES`

Nothing here re-implements the deterministic kernel; the runtime modules call
``retracemem.authorize(...)``. This module only describes and validates the
*data* that flows in and out of the learned components.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retracemem.schemas import AuthorizationStatus, DefeatPathType, EvidenceEdgeType

# Canonical typed revision action vocabulary (must match
# retracemem.proposers.typed_revision_policy.CANONICAL_ACTIONS).
CANONICAL_ACTIONS: frozenset[str] = frozenset(
    {"SUPERSEDES", "BLOCKS", "RELEASES", "UNCERTAIN", "REAFFIRMS", "NO_REVISION"}
)

# Actions whose target is a belief / condition (used for grounding checks).
BELIEF_TARGET_ACTIONS: frozenset[str] = frozenset({"SUPERSEDES", "UNCERTAIN", "REAFFIRMS"})
CONDITION_TARGET_ACTIONS: frozenset[str] = frozenset({"BLOCKS", "RELEASES"})

# Canonical DPA final-status vocabulary (matches retracemem AuthorizationStatus).
FINAL_STATUSES: frozenset[str] = frozenset(s.value for s in AuthorizationStatus)

# Candidate defeat-path types for the learned path ranker. The first three are
# the canonical DPA defeat paths; AUTHORIZED_DEFAULT is the no-defeat outcome.
CANDIDATE_PATH_TYPES: tuple[str, ...] = tuple(
    [p.value for p in DefeatPathType] + ["AUTHORIZED_DEFAULT"]
)


class SchemaValidationError(ValueError):
    """Raised when a training example violates the MemPatch scaffold schema."""


# ---------------------------------------------------------------------------
# Revision action (the unit emitted by the learned typed-revision proposer)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RevisionAction:
    """A single typed revision action.

    Field-level constraints (enforced by :meth:`validate`):

    * ``action_type`` in :data:`CANONICAL_ACTIONS`;
    * every action carries at least one grounding ``evidence_id``
      (including ``NO_REVISION``);
    * ``SUPERSEDES`` -> ``target_belief_id`` + ``replacement_belief_id``;
    * ``BLOCKS`` / ``RELEASES`` -> ``target_condition_id`` only;
    * ``UNCERTAIN`` / ``REAFFIRMS`` -> ``target_belief_id`` only;
    * ``NO_REVISION`` -> no belief/condition/replacement target.

    The v1 action object is closed-world: ``action_type`` from the canonical
    enum and every id slot drawn from the candidate graph / visible evidence.
    There is no open-ended ``scope`` field (the runtime parser and
    ``TypedRevisionTarget`` do not model one).
    """

    action_type: str
    target_belief_id: str | None = None
    target_condition_id: str | None = None
    replacement_belief_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target_belief_id": self.target_belief_id,
            "target_condition_id": self.target_condition_id,
            "replacement_belief_id": self.replacement_belief_id,
            "evidence_ids": list(self.evidence_ids),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RevisionAction":
        if "action_type" not in d:
            raise SchemaValidationError("action object missing 'action_type'")
        return cls(
            action_type=d["action_type"],
            target_belief_id=d.get("target_belief_id"),
            target_condition_id=d.get("target_condition_id"),
            replacement_belief_id=d.get("replacement_belief_id"),
            evidence_ids=tuple(d.get("evidence_ids", ()) or ()),
            rationale=d.get("rationale", "") or "",
        )

    def validate(self) -> None:
        if self.action_type not in CANONICAL_ACTIONS:
            raise SchemaValidationError(
                f"action_type '{self.action_type}' not in canonical vocabulary"
            )

        # Every action carries grounding evidence -- including NO_REVISION,
        # which means "after inspecting this new evidence, make no revision."
        # This matches the runtime parser, which requires every action to cite
        # the new evidence id.
        if not self.evidence_ids:
            raise SchemaValidationError(
                f"{self.action_type} requires at least one evidence_id"
            )

        if self.action_type == "NO_REVISION":
            if self.target_belief_id or self.target_condition_id or self.replacement_belief_id:
                raise SchemaValidationError(
                    "NO_REVISION must not carry target belief/condition/replacement"
                )
            return

        if self.action_type == "SUPERSEDES":
            if not self.target_belief_id:
                raise SchemaValidationError("SUPERSEDES requires target_belief_id")
            if not self.replacement_belief_id:
                raise SchemaValidationError("SUPERSEDES requires replacement_belief_id")
            if self.target_condition_id:
                raise SchemaValidationError("SUPERSEDES must not carry target_condition_id")
            if self.replacement_belief_id == self.target_belief_id:
                raise SchemaValidationError("SUPERSEDES replacement must differ from target")
        elif self.action_type in CONDITION_TARGET_ACTIONS:
            if not self.target_condition_id:
                raise SchemaValidationError(f"{self.action_type} requires target_condition_id")
            if self.target_belief_id or self.replacement_belief_id:
                raise SchemaValidationError(
                    f"{self.action_type} must only target a condition"
                )
        elif self.action_type in {"UNCERTAIN", "REAFFIRMS"}:
            if not self.target_belief_id:
                raise SchemaValidationError(f"{self.action_type} requires target_belief_id")
            if self.target_condition_id or self.replacement_belief_id:
                raise SchemaValidationError(
                    f"{self.action_type} must only target a belief"
                )

    @property
    def evidence_edge_type(self) -> EvidenceEdgeType | None:
        """Map to the runtime EvidenceEdgeType (None for NO_REVISION)."""
        if self.action_type == "NO_REVISION":
            return None
        return EvidenceEdgeType(self.action_type)


def validate_actions(actions: list[RevisionAction]) -> None:
    for a in actions:
        a.validate()


# ---------------------------------------------------------------------------
# Revision view (output of Scenario View Builder / input of Revision Response Policy)
# ---------------------------------------------------------------------------

_GRAPH_KEYS = (
    "evidence_nodes",
    "belief_nodes",
    "condition_nodes",
    "candidate_replacement_beliefs",
    "dependency_edges",
)


def validate_memory_graph(graph: dict[str, Any]) -> None:
    """Validate the structural shape of an extracted memory graph dict."""
    for key in _GRAPH_KEYS:
        if key not in graph:
            raise SchemaValidationError(f"memory graph missing required key '{key}'")
        if not isinstance(graph[key], list):
            raise SchemaValidationError(f"memory graph['{key}'] must be a list")

    evidence_ids = {e["evidence_id"] for e in graph["evidence_nodes"]}
    belief_ids = {b["belief_id"] for b in graph["belief_nodes"]}
    belief_ids |= {b["belief_id"] for b in graph["candidate_replacement_beliefs"]}
    condition_ids = {c["condition_id"] for c in graph["condition_nodes"]}

    for b in graph["belief_nodes"] + graph["candidate_replacement_beliefs"]:
        for ev in b.get("source_evidence_ids", []):
            if ev not in evidence_ids:
                raise SchemaValidationError(
                    f"belief '{b['belief_id']}' cites unknown evidence '{ev}'"
                )
    for edge in graph["dependency_edges"]:
        if edge.get("edge_type", "REQUIRES") != "REQUIRES":
            raise SchemaValidationError("dependency_edges only support edge_type REQUIRES")
        if edge["belief_id"] not in belief_ids:
            raise SchemaValidationError(
                f"dependency edge references unknown belief '{edge['belief_id']}'"
            )
        if edge["condition_id"] not in condition_ids:
            raise SchemaValidationError(
                f"dependency edge references unknown condition '{edge['condition_id']}'"
            )


# ---------------------------------------------------------------------------
# Training example schemas (the three JSONL row types)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphExtractionExample:
    """Row of ``graph_extraction_sft.jsonl`` (Scenario View Builder SFT).

    scenario event_trace / subagent submissions -> structured revision view.
    """

    example_id: str
    raw_dialogue: str
    subagent_roles: list[str]
    output_graph: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "raw_dialogue": self.raw_dialogue,
            "subagent_roles": list(self.subagent_roles),
            "output_graph": self.output_graph,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GraphExtractionExample":
        return cls(
            example_id=d["example_id"],
            raw_dialogue=d["raw_dialogue"],
            subagent_roles=list(d.get("subagent_roles", [])),
            output_graph=d["output_graph"],
            metadata=d.get("metadata", {}),
        )

    def validate(self) -> None:
        if not self.example_id:
            raise SchemaValidationError("GraphExtractionExample requires example_id")
        if not self.raw_dialogue:
            raise SchemaValidationError("GraphExtractionExample requires raw_dialogue")
        validate_memory_graph(self.output_graph)


@dataclass(frozen=True)
class TypedRevisionExample:
    """Row of ``typed_revision_sft.jsonl`` (Revision Response Policy SFT).

    revision view + new evidence -> gold benchmark-compatible typed patch actions.
    """

    example_id: str
    current_graph: dict[str, Any]
    new_evidence: dict[str, Any]
    candidate_beliefs: list[dict[str, Any]]
    candidate_replacement_beliefs: list[dict[str, Any]]
    candidate_conditions_by_belief: dict[str, list[dict[str, Any]]]
    dependency_edges_by_belief: dict[str, list[dict[str, Any]]]
    gold_actions: list[dict[str, Any]]
    gold_final_statuses: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "current_graph": self.current_graph,
            "new_evidence": self.new_evidence,
            "candidate_beliefs": self.candidate_beliefs,
            "candidate_replacement_beliefs": self.candidate_replacement_beliefs,
            "candidate_conditions_by_belief": self.candidate_conditions_by_belief,
            "dependency_edges_by_belief": self.dependency_edges_by_belief,
            "gold_actions": self.gold_actions,
            "gold_final_statuses": self.gold_final_statuses,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TypedRevisionExample":
        return cls(
            example_id=d["example_id"],
            current_graph=d["current_graph"],
            new_evidence=d["new_evidence"],
            candidate_beliefs=d.get("candidate_beliefs", []),
            candidate_replacement_beliefs=d.get("candidate_replacement_beliefs", []),
            candidate_conditions_by_belief=d.get("candidate_conditions_by_belief", {}),
            dependency_edges_by_belief=d.get("dependency_edges_by_belief", {}),
            gold_actions=d.get("gold_actions", []),
            gold_final_statuses=d.get("gold_final_statuses", {}),
            metadata=d.get("metadata", {}),
        )

    def gold_action_objects(self) -> list[RevisionAction]:
        return [RevisionAction.from_dict(a) for a in self.gold_actions]

    def validate(self) -> None:
        if not self.example_id:
            raise SchemaValidationError("TypedRevisionExample requires example_id")
        if "evidence_id" not in self.new_evidence:
            raise SchemaValidationError("new_evidence requires evidence_id")
        for status in self.gold_final_statuses.values():
            if status not in FINAL_STATUSES:
                raise SchemaValidationError(f"unknown gold final status '{status}'")
        actions = self.gold_action_objects()
        validate_actions(actions)
        # Grounding: every cited target/replacement/evidence must be visible.
        belief_ids = {b["belief_id"] for b in self.candidate_beliefs}
        belief_ids |= {b["belief_id"] for b in self.candidate_replacement_beliefs}
        condition_ids: set[str] = set()
        for conds in self.candidate_conditions_by_belief.values():
            condition_ids |= {c["condition_id"] for c in conds}
        evidence_ids = {self.new_evidence["evidence_id"]}
        evidence_ids |= {e["evidence_id"] for e in self.current_graph.get("evidence_nodes", [])}
        for a in actions:
            if a.target_belief_id and a.target_belief_id not in belief_ids:
                raise SchemaValidationError(
                    f"gold action targets unknown belief '{a.target_belief_id}'"
                )
            if a.replacement_belief_id and a.replacement_belief_id not in belief_ids:
                raise SchemaValidationError(
                    f"gold action references unknown replacement '{a.replacement_belief_id}'"
                )
            if a.target_condition_id and a.target_condition_id not in condition_ids:
                raise SchemaValidationError(
                    f"gold action targets unknown condition '{a.target_condition_id}'"
                )
            for ev in a.evidence_ids:
                if ev not in evidence_ids:
                    raise SchemaValidationError(
                        f"gold action cites unknown evidence '{ev}'"
                    )


@dataclass(frozen=True)
class RLRolloutExample:
    """Row of ``dpa_rl_rollouts.jsonl`` (benchmark-grounded feedback).

    A single sampled Revision Response Policy rollout scored through
    parser + RevisionGate + DPA. Benchmark-aligned metrics supply the
    preference / reward signal; DPA itself does not learn.
    """

    example_id: str
    prompt_input: str
    sampled_actions: list[dict[str, Any]]
    parser_result: dict[str, Any]
    gate_decisions: list[dict[str, Any]]
    dpa_final_statuses: dict[str, str]
    gold_final_statuses: dict[str, str]
    reward_breakdown: dict[str, float]
    total_reward: float
    failure_category: str
    audit_trace: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "prompt_input": self.prompt_input,
            "sampled_actions": self.sampled_actions,
            "parser_result": self.parser_result,
            "gate_decisions": self.gate_decisions,
            "dpa_final_statuses": self.dpa_final_statuses,
            "gold_final_statuses": self.gold_final_statuses,
            "reward_breakdown": self.reward_breakdown,
            "total_reward": self.total_reward,
            "failure_category": self.failure_category,
            "audit_trace": self.audit_trace,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RLRolloutExample":
        return cls(
            example_id=d["example_id"],
            prompt_input=d["prompt_input"],
            sampled_actions=d.get("sampled_actions", []),
            parser_result=d.get("parser_result", {}),
            gate_decisions=d.get("gate_decisions", []),
            dpa_final_statuses=d.get("dpa_final_statuses", {}),
            gold_final_statuses=d.get("gold_final_statuses", {}),
            reward_breakdown=d.get("reward_breakdown", {}),
            total_reward=d.get("total_reward", 0.0),
            failure_category=d.get("failure_category", "NONE"),
            audit_trace=d.get("audit_trace", {}),
            metadata=d.get("metadata", {}),
        )

    def validate(self) -> None:
        if not self.example_id:
            raise SchemaValidationError("RLRolloutExample requires example_id")
        for status in list(self.dpa_final_statuses.values()) + list(
            self.gold_final_statuses.values()
        ):
            if status not in FINAL_STATUSES:
                raise SchemaValidationError(f"unknown final status '{status}'")

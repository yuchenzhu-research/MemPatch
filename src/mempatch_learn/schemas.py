"""MemPatch Revision Module training-data contracts.

JSONL schemas for the revision module pipeline (Algorithm 1 in
``AGENTS.md``): scenario/event_trace → revision view
→ benchmark-compatible response → DPA-Consistent Projection →
benchmark-grounded feedback. Aligns with ``mempatch_dpa`` runtime vocabulary:

* typed action vocabulary  -> :data:`CANONICAL_ACTIONS`
  (``SUPERSEDES``/``BLOCKS``/``RELEASES``/``UNCERTAIN``/``REAFFIRMS``/``NO_REVISION``)
* DPA final status vocabulary -> :data:`FINAL_STATUSES`
  (``AUTHORIZED``/``SUPERSEDED``/``BLOCKED``/``UNRESOLVED``)
* candidate defeat-path types -> :data:`CANDIDATE_PATH_TYPES`

Nothing here re-implements the deterministic kernel; the runtime modules call
``mempatch_dpa.authorize(...)``. This module only describes and validates the
*data* that flows in and out of the learned components.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mempatch_dpa.schemas import AuthorizationStatus, DefeatPathType, EvidenceEdgeType

# Canonical typed revision action vocabulary (shared with the DPA action parser).
CANONICAL_ACTIONS: frozenset[str] = frozenset(
    {"SUPERSEDES", "BLOCKS", "RELEASES", "UNCERTAIN", "REAFFIRMS", "NO_REVISION"}
)

# Actions whose target is a belief / condition (used for grounding checks).
BELIEF_TARGET_ACTIONS: frozenset[str] = frozenset({"SUPERSEDES", "UNCERTAIN", "REAFFIRMS"})
CONDITION_TARGET_ACTIONS: frozenset[str] = frozenset({"BLOCKS", "RELEASES"})

# Canonical DPA final-status vocabulary (matches mempatch_dpa AuthorizationStatus).
FINAL_STATUSES: frozenset[str] = frozenset(s.value for s in AuthorizationStatus)

# Candidate defeat-path types for the learned path ranker. The first three are
# the canonical DPA defeat paths; AUTHORIZED_DEFAULT is the no-defeat outcome.
CANDIDATE_PATH_TYPES: tuple[str, ...] = tuple(
    [p.value for p in DefeatPathType] + ["AUTHORIZED_DEFAULT"]
)


class SchemaValidationError(ValueError):
    """Raised when a training example violates the Revision Module schema."""


@dataclass(frozen=True)
class RevisionAction:
    """A single typed revision action emitted by the learned proposer."""

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
        if self.action_type == "NO_REVISION":
            return None
        return EvidenceEdgeType(self.action_type)


def validate_actions(actions: list[RevisionAction]) -> None:
    for action in actions:
        action.validate()

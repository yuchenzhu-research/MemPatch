"""Data contracts for the raw-dialogue protocol and graph extraction task.

These schemas define the input and output structures for processing raw dialogue
into structured candidate memory graphs prior to revision authorization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class RawDialogueValidationError(ValueError):
    """Raised when raw dialogue data structures violate schema constraints."""


@dataclass(frozen=True)
class RawDialogueUtterance:
    """A single utterance in a subagent interaction dialogue."""

    speaker: str
    text: str
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "text": self.text,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RawDialogueUtterance":
        if "speaker" not in d or "text" not in d:
            raise RawDialogueValidationError("Utterance must carry 'speaker' and 'text'")
        return cls(
            speaker=d["speaker"],
            text=d["text"],
            timestamp=d.get("timestamp"),
            metadata=d.get("metadata", {}),
        )


@dataclass(frozen=True)
class RawDialogue:
    """Sequence of subagent utterances representing the raw input log."""

    utterances: tuple[RawDialogueUtterance, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "utterances": [u.to_dict() for u in self.utterances],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RawDialogue":
        if "utterances" not in d:
            raise RawDialogueValidationError("RawDialogue must contain 'utterances'")
        return cls(
            utterances=tuple(RawDialogueUtterance.from_dict(u) for u in d["utterances"]),
            metadata=d.get("metadata", {}),
        )


@dataclass(frozen=True)
class DialogueExtractionTarget:
    """The target structures extracted from a raw dialogue sequence."""

    example_id: str
    dialogue: RawDialogue
    subagent_roles: list[str]
    gold_graph: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "dialogue": self.dialogue.to_dict(),
            "subagent_roles": list(self.subagent_roles),
            "gold_graph": self.gold_graph,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DialogueExtractionTarget":
        if "example_id" not in d or "dialogue" not in d or "gold_graph" not in d:
            raise RawDialogueValidationError("DialogueExtractionTarget missing required fields")
        return cls(
            example_id=d["example_id"],
            dialogue=RawDialogue.from_dict(d["dialogue"]),
            subagent_roles=list(d.get("subagent_roles", [])),
            gold_graph=d["gold_graph"],
            metadata=d.get("metadata", {}),
        )

    def validate(self) -> None:
        """Validate structural completeness and key mapping constraints."""
        if not self.example_id:
            raise RawDialogueValidationError("example_id must not be empty")
        if not self.dialogue.utterances:
            raise RawDialogueValidationError("dialogue must carry at least one utterance")

        graph = self.gold_graph
        required_keys = {
            "evidence_nodes",
            "belief_nodes",
            "condition_nodes",
            "candidate_replacement_beliefs",
            "dependency_edges",
        }
        for k in required_keys:
            if k not in graph:
                raise RawDialogueValidationError(f"gold_graph missing required key '{k}'")
            if not isinstance(graph[k], list):
                raise RawDialogueValidationError(f"gold_graph['{k}'] must be a list")

        evidence_ids = {e["evidence_id"] for e in graph["evidence_nodes"] if "evidence_id" in e}
        belief_ids = {b["belief_id"] for b in graph["belief_nodes"] if "belief_id" in b}
        belief_ids |= {b["belief_id"] for b in graph["candidate_replacement_beliefs"] if "belief_id" in b}
        condition_ids = {c["condition_id"] for c in graph["condition_nodes"] if "condition_id" in c}

        for b in graph["belief_nodes"] + graph["candidate_replacement_beliefs"]:
            for ev_id in b.get("source_evidence_ids", []):
                if ev_id not in evidence_ids:
                    raise RawDialogueValidationError(
                        f"belief '{b.get('belief_id')}' references unknown evidence '{ev_id}'"
                    )

        for edge in graph["dependency_edges"]:
            if edge.get("edge_type", "REQUIRES") != "REQUIRES":
                raise RawDialogueValidationError("dependency_edges only support edge_type REQUIRES")
            if edge.get("belief_id") not in belief_ids:
                raise RawDialogueValidationError(
                    f"dependency edge references unknown belief '{edge.get('belief_id')}'"
                )
            if edge.get("condition_id") not in condition_ids:
                raise RawDialogueValidationError(
                    f"dependency edge references unknown condition '{edge.get('condition_id')}'"
                )

"""Structured error contracts for the MemPatch commit path (ReTrace-Engine).

Every stage of the deterministic backend (Parser, RevisionGate, DPA) reports
failures and warnings through a shared ``EngineError`` dataclass. This enables:

* structured reward shaping (penalties proportional to error severity)
* auditable fail-closed behavior (every rejection carries a machine-readable code)
* curriculum-driven training (errors are categorized for analysis)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EngineStage(str, Enum):
    PARSER = "PARSER"
    REVISION_GATE = "REVISION_GATE"
    DPA = "DPA"


class ErrorSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class EngineError:
    """A single structured error or warning from an engine stage."""

    stage: EngineStage
    code: str
    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    fail_closed: bool = True
    action_index: int | None = None
    belief_id: str | None = None
    condition_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "fail_closed": self.fail_closed,
            "action_index": self.action_index,
            "belief_id": self.belief_id,
            "condition_id": self.condition_id,
            "evidence_ids": list(self.evidence_ids),
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Canonical error codes per stage
# ---------------------------------------------------------------------------

# Parser error codes
PARSER_INVALID_JSON = "PARSER_INVALID_JSON"
PARSER_UNKNOWN_ACTION_TYPE = "PARSER_UNKNOWN_ACTION_TYPE"
PARSER_MISSING_REQUIRED_FIELD = "PARSER_MISSING_REQUIRED_FIELD"
PARSER_SCHEMA_VIOLATION = "PARSER_SCHEMA_VIOLATION"
PARSER_ITEM_NOT_OBJECT = "PARSER_ITEM_NOT_OBJECT"

# Gate error codes
GATE_UNKNOWN_BELIEF = "GATE_UNKNOWN_BELIEF"
GATE_UNKNOWN_CONDITION = "GATE_UNKNOWN_CONDITION"
GATE_MISSING_EVIDENCE = "GATE_MISSING_EVIDENCE"
GATE_MISSING_REPLACEMENT = "GATE_MISSING_REPLACEMENT"
GATE_REPLACEMENT_EQUALS_TARGET = "GATE_REPLACEMENT_EQUALS_TARGET"
GATE_TARGET_KIND_MISMATCH = "GATE_TARGET_KIND_MISMATCH"
GATE_MISSING_PROVENANCE = "GATE_MISSING_PROVENANCE"
GATE_INVALID_SCOPE_TYPE = "GATE_INVALID_SCOPE_TYPE"
GATE_RELEASE_WITHOUT_PRIOR_BLOCK = "GATE_RELEASE_WITHOUT_PRIOR_BLOCK"
GATE_REPLACEMENT_NOT_IN_STORE = "GATE_REPLACEMENT_NOT_IN_STORE"

# DPA warning codes
DPA_NO_EDGES_FOR_BELIEF = "DPA_NO_EDGES_FOR_BELIEF"
DPA_MISSING_EVIDENCE_ATOM = "DPA_MISSING_EVIDENCE_ATOM"

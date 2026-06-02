"""Schema helpers for the general English ReTrace-Bench release.

The benchmark files are JSONL dictionaries for portability. These dataclasses
document the expected shape and make tests/imports explicit without forcing a
heavy runtime dependency.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GeneralMemoryEntry:
    memory_id: str
    text: str
    visibility_scope: str
    source_event_ids: list[str] = field(default_factory=list)
    introduced_by_event_id: str | None = None
    is_distractor: bool = False


@dataclass(frozen=True)
class GeneralEvent:
    event_id: str
    timestamp: str
    source: str
    actor: str
    event_type: str
    text: str
    trust_level: str
    visibility_scope: str
    related_memory_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GeneralTask:
    task_id: str
    task_type: str
    prompt: str


@dataclass(frozen=True)
class GeneralHiddenGold:
    expected_answer: str
    expected_decision: str | None
    expected_evidence_event_ids: list[str]
    expected_memory_state: dict[str, str]
    expected_failure_diagnosis: str
    stale_or_wrong_answers: list[str]
    rubric: dict[str, Any]


@dataclass(frozen=True)
class GeneralScenario:
    scenario_id: str
    domain: str
    primary_failure_mode: str
    secondary_failure_modes: list[str]
    difficulty: str
    workflow_context: str
    public_input: dict[str, Any]
    tasks: list[GeneralTask]
    hidden_gold: GeneralHiddenGold
    metadata: dict[str, Any]


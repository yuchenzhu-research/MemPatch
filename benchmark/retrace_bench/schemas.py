from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from benchmark.retrace_bench.taxonomy import Domain, RevisionFamily, RevisionActionType, FinalStatus, ProbeType


@dataclass(frozen=True)
class DialogueTurn:
    speaker: str
    text: str
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryEntry:
    entry_id: str
    content: str
    entry_type: str  # "belief", "condition", "evidence"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RevisionAction:
    action_type: RevisionActionType
    target_id: str
    replacement_id: Optional[str] = None
    evidence_ids: List[str] = field(default_factory=list)
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeQuery:
    query_id: str
    probe_type: ProbeType
    question: str
    options: Dict[str, str]  # e.g., {"A": "...", "B": "..."}
    gold_answer: str        # e.g., "A"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    domain: Domain
    revision_family: RevisionFamily
    conflict_type: str
    memory_topology: Dict[str, Any]  # e.g., {"requires": {belief_id: [cond_id]}}
    dialogue_history: List[DialogueTurn]
    memory_snapshot: List[MemoryEntry]
    gold_final_statuses: Dict[str, FinalStatus]  # memory_id -> status
    gold_revision_actions: List[RevisionAction]
    probe_queries: List[ProbeQuery]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    scenario_id: str
    query_id: str
    predicted_answer: str
    predicted_final_statuses: Optional[Dict[str, FinalStatus]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    scenario_id: str
    query_id: str
    probe_type: ProbeType
    is_correct: bool
    status_accuracy: Optional[float] = None
    audit_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Manifest:
    version: str
    num_scenarios: int
    num_queries: int
    domains: List[str]
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationReport:
    is_valid: bool
    errors: List[str]
    num_checked: int
    metadata: Dict[str, Any] = field(default_factory=dict)

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from benchmark.retrace_bench.schemas import DialogueTurn, MemoryEntry, RevisionAction
from benchmark.retrace_bench.taxonomy import FinalStatus


@dataclass(frozen=True)
class StructuredProtocolInput:
    dialogue_history: List[DialogueTurn]
    memory_snapshot: List[MemoryEntry]
    candidate_graph: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class StructuredProtocolOutput:
    revision_actions: List[RevisionAction]
    final_statuses: Optional[Dict[str, FinalStatus]] = None
    rationale: Optional[str] = None

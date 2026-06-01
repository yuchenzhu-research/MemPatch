from dataclasses import dataclass, field
from typing import Dict, List, Optional
from benchmark.retrace_bench.schemas import DialogueTurn, MemoryEntry, ProbeQuery
from benchmark.retrace_bench.taxonomy import FinalStatus


@dataclass(frozen=True)
class RawProtocolInput:
    dialogue_history: List[DialogueTurn]
    memory_snapshot: List[MemoryEntry]
    query: ProbeQuery


@dataclass(frozen=True)
class RawProtocolOutput:
    answer: str
    final_statuses: Optional[Dict[str, FinalStatus]] = None
    rationale: Optional[str] = None

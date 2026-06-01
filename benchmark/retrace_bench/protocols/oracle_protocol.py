from dataclasses import dataclass
from typing import Any, Dict
from benchmark.retrace_bench.taxonomy import FinalStatus


@dataclass(frozen=True)
class OracleProtocolInput:
    gold_graph: Dict[str, Any]
    gold_candidate_structure: Dict[str, Any]
    oracle_memory_links: Dict[str, Any]


@dataclass(frozen=True)
class OracleProtocolOutput:
    final_statuses: Dict[str, FinalStatus]
    rationale: str | None = None

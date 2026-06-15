"""Reference Transition Semantics: Configuration and State definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class MemoryState:
    memory_id: str
    status: str  # current, blocked, should_not_store, unresolved, out_of_scope
    text: str
    scope: str

@dataclass(frozen=True)
class Configuration:
    memories: dict[str, MemoryState]  # memory_id -> MemoryState
    t: int  # step index/timestamp
    trace: list[dict[str, Any]] = field(default_factory=list)

    def get_status_dict(self) -> dict[str, str]:
        return {mid: m.status for mid, m in self.memories.items()}

    def copy_with(self, *, memories: dict[str, MemoryState] | None = None, t: int | None = None, trace_step: dict[str, Any] | None = None) -> Configuration:
        new_m = memories if memories is not None else self.memories
        new_t = t if t is not None else self.t
        new_trace = list(self.trace)
        if trace_step is not None:
            new_trace.append(trace_step)
        return Configuration(memories=new_m, t=new_t, trace=new_trace)

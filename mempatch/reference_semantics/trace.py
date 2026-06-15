"""Reference Transition Semantics: Trace structure and validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class DerivationTrace:
    scenario_id: str
    steps: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "steps": self.steps,
        }

"""Reference Transition Semantics: Actions definitions."""
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class TransitionAction:
    action_type: str  # MUTATE_STATUS, NOOP
    target_record_id: str
    target_status: str
    rationale: str

"""Reference Transition Semantics: Priority and Precedence Resolution."""
from __future__ import annotations

from typing import Any
from mempatch.reference_semantics.rules import TransitionRule

def resolve_priority(matched_rules_with_events: list[tuple[TransitionRule, dict[str, Any]]]) -> tuple[TransitionRule, dict[str, Any]] | None:
    """Select the rule application with the highest priority (lowest priority integer).

    If there is a tie, tie-break deterministically using the event timestamp or order.
    """
    if not matched_rules_with_events:
        return None

    # Sort by priority ascending (REFUSE < ESCALATE < ASK < MARK < USE)
    # Then sort by timestamp_order ascending to prioritize earlier events in tie-breaks
    sorted_matches = sorted(
        matched_rules_with_events,
        key=lambda item: (item[0].priority, item[1].get("timestamp_order", 0))
    )
    return sorted_matches[0]

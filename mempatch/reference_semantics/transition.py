"""Reference Transition Semantics: Transition Engine implementation."""
from __future__ import annotations

from typing import Any
from mempatch.reference_semantics.states import Configuration, MemoryState
from mempatch.reference_semantics.rules import RULES, TransitionRule

class ReferenceTransitionEngine:
    """Transition engine that steps through events and updates the configuration."""

    def __init__(self, initial_config: Configuration) -> None:
        self.config = initial_config

    def step(self, event: dict[str, Any]) -> Configuration:
        """Process a single event, potentially causing a state transition.

        We check all rules against the event. If a rule matches and its priority is higher
        (i.e., lower priority number) than any rule matched so far, we perform a state transition.
        """
        event_text = event.get("text", "")
        event_id = event.get("event_id", "")
        
        # Check all rules
        matched_rules = [r for r in RULES if r.matches(event_text)]
        if not matched_rules:
            # No matching rule, t advances but state remains same
            return self.config.copy_with(t=self.config.t + 1)

        # Sort matched rules by priority (highest first, i.e., lowest priority number)
        matched_rules = sorted(matched_rules, key=lambda r: r.priority)
        best_rule = matched_rules[0]

        # Find target memory
        # In this benchmark, the mutable record is the 'target' memory
        target_mid = None
        target_pre_state = "current"
        for mid, mem in self.config.memories.items():
            if mid.endswith("-target") or "target" in mid:
                target_mid = mid
                target_pre_state = mem.status
                break

        if target_mid is None:
            # Fallback to first memory
            target_mid = list(self.config.memories.keys())[0]
            target_pre_state = self.config.memories[target_mid].status

        # Evaluate if we should transition:
        # We transition if the new rule has HIGHER priority (lower priority number) than the target's current state.
        # Status priority hierarchy mapping:
        # should_not_store (1) > blocked (2,3) > unresolved (4) > current (5)
        status_priority_map = {
            "should_not_store": 1,
            "blocked": 2,
            "unresolved": 4,
            "current": 5,
            "out_of_scope": 99,
        }

        current_pri = status_priority_map.get(target_pre_state, 5)
        new_pri = best_rule.priority

        # We transition if the new rule's priority is strictly higher
        # Or if the new rule has the same priority but provides a different transition reason/code
        if new_pri < current_pri or (new_pri == current_pri and target_pre_state == "current" and best_rule.target_status != "current"):
            # Update memories
            new_memories = dict(self.config.memories)
            old_mem = new_memories[target_mid]
            new_memories[target_mid] = MemoryState(
                memory_id=target_mid,
                status=best_rule.target_status,
                text=old_mem.text,
                scope=old_mem.scope
            )

            # Record step trace
            trace_step = {
                "rule_id": best_rule.rule_id,
                "target_record_id": target_mid,
                "evidence_ids": [event_id],
                "pre_state": target_pre_state,
                "post_state": best_rule.target_status,
                "priority": best_rule.priority,
                "reason_code": best_rule.reason_code,
            }
            return self.config.copy_with(memories=new_memories, t=self.config.t + 1, trace_step=trace_step)

        return self.config.copy_with(t=self.config.t + 1)

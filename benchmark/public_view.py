"""Sanitized public views for model-facing benchmark runners."""

from __future__ import annotations

import copy
from typing import Any

# Fields that must never appear in model prompts or public exports.
INTERNAL_ONLY_FIELDS = frozenset(
    {
        "is_distractor",
        "hidden_gold",
        "validation_notes",
        "metadata",
        "source_pointers",
        "primary_failure_mode",
        "pattern_trap_type",
        "canonical_failure_mode",
    }
)

# Allowed keys on event objects in the public view.
PUBLIC_EVENT_KEYS = frozenset(
    {
        "event_id",
        "timestamp_order",
        "actor_role",
        "trust_level",
        "visibility_scope",
        "event_type",
        "text",
        "related_memory_ids",
        "timestamp",
    }
)

# Allowed keys on initial_memory entries in the public view.
PUBLIC_MEMORY_KEYS = frozenset(
    {
        "memory_id",
        "text",
        "scope",
        "source_event_ids",
    }
)


def _strip_internal(obj: Any) -> Any:
    """Recursively deep-copy and remove internal-only keys from dicts."""
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for key, value in obj.items():
            if key in INTERNAL_ONLY_FIELDS:
                continue
            cleaned[key] = _strip_internal(value)
        return cleaned
    if isinstance(obj, list):
        return [_strip_internal(item) for item in obj]
    return copy.deepcopy(obj)


def sanitize_public_input(public_input: dict[str, Any] | None) -> dict[str, Any]:
    """Return a deep copy of public_input safe for model consumption."""
    if not public_input:
        return {}
    sanitized = _strip_internal(public_input)
    events = sanitized.get("event_trace")
    if isinstance(events, list):
        sanitized["event_trace"] = [
            {k: ev[k] for k in PUBLIC_EVENT_KEYS if k in ev} for ev in events if isinstance(ev, dict)
        ]
    memories = sanitized.get("initial_memory")
    if isinstance(memories, list):
        sanitized["initial_memory"] = [
            {k: mem[k] for k in PUBLIC_MEMORY_KEYS if k in mem} for mem in memories if isinstance(mem, dict)
        ]
    return sanitized


def public_scenario_view(scenario: dict[str, Any]) -> dict[str, Any]:
    """Build the model-visible scenario payload without internal leakage fields."""
    tasks: dict[str, Any] = {}
    for key in ("black_box_task", "memory_state_task", "evidence_retrieval_task", "diagnostic_task"):
        if key in scenario:
            tasks[key] = _strip_internal(scenario[key])
    if not tasks and scenario.get("tasks"):
        raw_tasks = scenario["tasks"]
        if isinstance(raw_tasks, dict):
            tasks = _strip_internal(raw_tasks)
        elif isinstance(raw_tasks, list):
            tasks = {"tasks": _strip_internal(raw_tasks)}

    view: dict[str, Any] = {
        "scenario_id": scenario["scenario_id"],
        "domain": scenario.get("domain"),
        "difficulty": scenario.get("difficulty") or scenario.get("difficulty_level"),
        "workflow_context": _strip_internal(scenario.get("workflow_context", "")),
        "public_input": sanitize_public_input(scenario.get("public_input", {})),
    }
    view.update(_strip_internal(tasks))
    return view

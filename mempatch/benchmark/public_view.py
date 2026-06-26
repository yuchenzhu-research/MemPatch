"""Sanitized public views for model-facing benchmark runners."""

from __future__ import annotations

import copy
from typing import Any

from mempatch.benchmark.leakage import sanitize_public_value

# Fields that must never appear in model prompts or public exports.
INTERNAL_ONLY_FIELDS = frozenset(
    {
        "canonical_failure_mode",
        "decision_triggers",
        "difficulty",
        "difficulty_level",
        "expected_answer",
        "expected_decision",
        "expected_evidence_event_ids",
        "expected_failure_diagnosis",
        "expected_memory_state",
        "expected_memory_states",
        "failure_mode",
        "generation_metadata",
        "is_distractor",
        "hidden_gold",
        "metadata",
        "pattern",
        "pattern_trap_type",
        "primary_failure_mode",
        "resolver_trace",
        "source_pointers",
        "template_family_id",
        "template_instance_id",
        "validation_notes",
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
        "content",
        "related_memory_ids",
        "timestamp",
        "source",
    }
)

# Allowed keys on initial_memory entries in the public view.
PUBLIC_MEMORY_KEYS = frozenset(
    {
        "memory_id",
        "text",
        "content",
        "scope",
        "source_event_ids",
        "memory_type",
        "user_id",
        "session_id",
        "owner_id",
        "created_at",
        "category",
        "tags",
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


def _sanitize_item(item: dict[str, Any], allowed_keys: frozenset[str]) -> dict[str, Any]:
    cleaned = {
        key: sanitize_public_value(item[key])
        for key in allowed_keys
        if key in item
    }
    return {
        key: value
        for key, value in cleaned.items()
        if value is not None and value != [] and value != {}
    }


def sanitize_public_input(public_input: dict[str, Any] | None) -> dict[str, Any]:
    """Return a deep copy of public_input safe for model consumption."""
    if not public_input:
        return {}
    sanitized = _strip_internal(public_input)
    events = sanitized.get("events", sanitized.get("event_trace"))
    if isinstance(events, list):
        sanitized["events"] = [
            _sanitize_item(ev, PUBLIC_EVENT_KEYS) for ev in events if isinstance(ev, dict)
        ]
        sanitized["event_trace"] = sanitized["events"]
    memories = sanitized.get("initial_memories", sanitized.get("initial_memory"))
    if isinstance(memories, list):
        sanitized["initial_memories"] = [
            _sanitize_item(mem, PUBLIC_MEMORY_KEYS) for mem in memories if isinstance(mem, dict)
        ]
        sanitized["initial_memory"] = sanitized["initial_memories"]
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
        "workflow_context": _strip_internal(scenario.get("workflow_context", "")),
        "public_input": sanitize_public_input(scenario.get("public_input", {})),
    }
    view.update(sanitize_public_value(_strip_internal(tasks)))
    return view

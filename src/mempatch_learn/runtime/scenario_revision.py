"""Scenario View Builder adapter for MemPatch-Bench scenarios (Algorithm Step 1)."""
from __future__ import annotations

import re
from typing import Any

from mempatch_dpa.methods.contracts import SharedCandidateView

from mempatch_learn.runtime.views import build_view

_REPLACEMENT_KEYWORDS = (
    "update",
    "updated",
    "supersedes",
    "replaces",
    "no longer",
    "changed to",
    "now",
    "instead",
    "corrected",
    "reverted",
)

_CONDITION_KEYWORD_RE = re.compile(
    r"\b("
    r"if|unless|until|after|before|while|"
    r"only if|condition|temporary|hold|block|release"
    r")\b",
    re.IGNORECASE,
)

_VIEW_BUILDER_INDUCER = "scenario_view_builder_v1"


def _task_prompt(scenario: dict[str, Any]) -> str:
    for key in (
        "black_box_task",
        "memory_state_task",
        "evidence_retrieval_task",
        "diagnostic_task",
    ):
        task = scenario.get(key) or {}
        prompt = task.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt
    return scenario.get("workflow_context", "") or ""


def _event_suggests_replacement(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in _REPLACEMENT_KEYWORDS)


def _memory_has_condition_text(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith("Condition rule:"):
        return True
    return _CONDITION_KEYWORD_RE.search(stripped) is not None


def _paired_target_for_condition_memory(
    memory_id: str,
    memory_by_id: dict[str, dict[str, Any]],
) -> str | None:
    suffix_pairs = (
        ("-condition", "-target"),
        ("_condition", "_target"),
        ("_condition", "-target"),
    )
    for suffix, target_suffix in suffix_pairs:
        if memory_id.endswith(suffix):
            candidate = f"{memory_id[: -len(suffix)]}{target_suffix}"
            if candidate in memory_by_id:
                return candidate
    return None


def _extract_replacement_candidates(
    events: list[dict[str, Any]],
    *,
    known_memory_ids: set[str],
) -> list[dict[str, Any]]:
    replacements: list[dict[str, Any]] = []
    seen_replacement_ids: set[str] = set()

    for event in events:
        text = str(event.get("text") or "").strip()
        if not text or not _event_suggests_replacement(text):
            continue
        event_id = str(event["event_id"])
        related_ids = event.get("related_memory_ids") or []
        for raw_memory_id in related_ids:
            memory_id = str(raw_memory_id)
            if memory_id not in known_memory_ids:
                continue
            replacement_id = f"{memory_id}__replacement__{event_id}"
            if replacement_id in seen_replacement_ids:
                continue
            seen_replacement_ids.add(replacement_id)
            replacements.append(
                {
                    "belief_id": replacement_id,
                    "proposition": text,
                    "source_evidence_ids": [event_id],
                }
            )
    return replacements


def _append_condition_and_dependency(
    *,
    belief_id: str,
    condition_text: str,
    scope_id: str,
    condition_suffix: str,
    conditions_by_belief: dict[str, list[dict[str, Any]]],
    dependency_edges_by_belief: dict[str, list[dict[str, Any]]],
    seen_condition_ids: set[str],
) -> None:
    condition_id = f"{scope_id}__cond__{condition_suffix}"
    if condition_id in seen_condition_ids:
        return
    seen_condition_ids.add(condition_id)
    conditions_by_belief.setdefault(belief_id, []).append(
        {
            "condition_id": condition_id,
            "scope_id": scope_id,
            "text": condition_text.strip(),
        }
    )
    dependency_edges_by_belief.setdefault(belief_id, []).append(
        {
            "edge_id": f"dep_{belief_id}_{condition_id}",
            "belief_id": belief_id,
            "condition_id": condition_id,
            "edge_type": "REQUIRES",
            "inducer": _VIEW_BUILDER_INDUCER,
        }
    )


def _extract_conditions_and_dependencies(
    memories: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    known_memory_ids: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    conditions_by_belief: dict[str, list[dict[str, Any]]] = {}
    dependency_edges_by_belief: dict[str, list[dict[str, Any]]] = {}
    seen_condition_ids: set[str] = set()

    memory_by_id = {
        str(memory["memory_id"]): memory
        for memory in memories
        if memory.get("memory_id")
    }

    for memory_id, memory in memory_by_id.items():
        text = str(memory.get("text") or "")
        if not _memory_has_condition_text(text):
            continue
        paired_target = _paired_target_for_condition_memory(memory_id, memory_by_id)
        belief_id = paired_target if paired_target in memory_by_id else memory_id
        _append_condition_and_dependency(
            belief_id=belief_id,
            condition_text=text,
            scope_id=memory_id,
            condition_suffix="0",
            conditions_by_belief=conditions_by_belief,
            dependency_edges_by_belief=dependency_edges_by_belief,
            seen_condition_ids=seen_condition_ids,
        )

    for event in events:
        text = str(event.get("text") or "").strip()
        if not text or not _memory_has_condition_text(text):
            continue
        if text.startswith("Condition rule:"):
            continue
        related_ids = event.get("related_memory_ids") or []
        if not related_ids:
            continue
        event_id = str(event["event_id"])
        for raw_memory_id in related_ids:
            memory_id = str(raw_memory_id)
            if memory_id not in known_memory_ids:
                continue
            _append_condition_and_dependency(
                belief_id=memory_id,
                condition_text=text,
                scope_id=memory_id,
                condition_suffix=f"event_{event_id}",
                conditions_by_belief=conditions_by_belief,
                dependency_edges_by_belief=dependency_edges_by_belief,
                seen_condition_ids=seen_condition_ids,
            )

    return conditions_by_belief, dependency_edges_by_belief


def build_scenario_revision_view(scenario: dict[str, Any]) -> SharedCandidateView:
    """``V ← BuildScenarioRevisionView(S, M)`` for a MemPatch-Bench scenario."""
    sid = str(scenario["scenario_id"])
    public = scenario.get("public_input") or {}
    events = list(public.get("event_trace") or [])
    memories = list(public.get("initial_memory") or [])
    if not events:
        raise ValueError(f"{sid}: public_input.event_trace is empty")

    sorted_events = sorted(
        events,
        key=lambda e: (str(e.get("timestamp") or ""), str(e.get("event_id") or "")),
    )
    new_evidence_id = str(sorted_events[-1]["event_id"])

    evidence_context = [
        {
            "evidence_id": str(e["event_id"]),
            "text": e.get("text", ""),
            "timestamp": e.get("timestamp"),
        }
        for e in events
    ]
    candidate_beliefs = [
        {
            "belief_id": str(m["memory_id"]),
            "proposition": m.get("text", ""),
            "source_evidence_ids": list(m.get("source_event_ids") or []),
        }
        for m in memories
        if m.get("memory_id")
    ]
    known_memory_ids = {belief["belief_id"] for belief in candidate_beliefs}
    candidate_replacement_beliefs = _extract_replacement_candidates(
        events,
        known_memory_ids=known_memory_ids,
    )
    candidate_conditions_by_belief, dependency_edges_by_belief = (
        _extract_conditions_and_dependencies(
            memories,
            events,
            known_memory_ids=known_memory_ids,
        )
    )

    return build_view(
        instance_id=sid,
        query_id=sid,
        query=_task_prompt(scenario),
        evidence_context=evidence_context,
        new_evidence_id=new_evidence_id,
        candidate_beliefs=candidate_beliefs,
        candidate_replacement_beliefs=candidate_replacement_beliefs,
        candidate_conditions_by_belief=candidate_conditions_by_belief,
        dependency_edges_by_belief=dependency_edges_by_belief,
    )

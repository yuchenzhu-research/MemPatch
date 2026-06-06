"""Scenario View Builder adapter for MemPatch-Bench scenarios (Algorithm Step 1)."""
from __future__ import annotations

from typing import Any

from retracemem.methods.contracts import SharedCandidateView

from retrace_learn.runtime.views import build_view


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

    return build_view(
        instance_id=sid,
        query_id=sid,
        query=_task_prompt(scenario),
        evidence_context=evidence_context,
        new_evidence_id=new_evidence_id,
        candidate_beliefs=candidate_beliefs,
        candidate_replacement_beliefs=[],
        candidate_conditions_by_belief={},
        dependency_edges_by_belief={},
    )

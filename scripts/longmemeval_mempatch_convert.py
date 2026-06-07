"""Convert LongMemEval instances into MemPatch-Bench scenarios for revision."""

from __future__ import annotations

import json
import re
from typing import Any

_UPDATE_KEYWORDS = (
    "update",
    "updated",
    "now",
    "changed",
    "recently",
    "instead",
    "no longer",
    "moved to",
    "switched to",
)


def _answer_turns(example: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    triples = sorted(
        zip(
            example.get("haystack_dates") or [],
            example.get("haystack_sessions") or [],
            example.get("haystack_session_ids") or [],
        ),
        key=lambda item: item[0],
    )
    turns: list[tuple[str, str, dict[str, Any]]] = []
    for date, session, session_id in triples:
        for turn in session:
            if turn.get("has_answer"):
                turns.append((date, session_id, turn))
    return turns


def _ensure_update_markers(text: str) -> str:
    lowered = text.lower()
    if any(keyword in lowered for keyword in _UPDATE_KEYWORDS):
        return text
    return f"Updated user information: {text}"


def convert_knowledge_update(example: dict[str, Any]) -> dict[str, Any]:
    """Map a LongMemEval knowledge-update item to a MemPatch scenario."""
    question_id = str(example["question_id"])
    if question_id.endswith("_abs"):
        raise ValueError(f"{question_id}: abstention items are not supported yet")

    memory_id = f"m-{question_id}-belief"
    answer_turns = _answer_turns(example)
    if len(answer_turns) < 2:
        raise ValueError(f"{question_id}: expected at least two evidence turns")

    old_date, old_session_id, old_turn = answer_turns[0]
    new_date, new_session_id, new_turn = answer_turns[-1]

    triples = sorted(
        zip(
            example.get("haystack_dates") or [],
            example.get("haystack_sessions") or [],
            example.get("haystack_session_ids") or [],
        ),
        key=lambda item: item[0],
    )

    events: list[dict[str, Any]] = []
    order = 1
    first_event_id: str | None = None
    for date, session, session_id in triples:
        for turn_index, turn in enumerate(session):
            event_id = f"e-{question_id}-{session_id}-{turn_index}"
            if first_event_id is None:
                first_event_id = event_id
            related_memory_ids: list[str] = []
            text = str(turn.get("content") or "")
            if (date, session_id, turn) == (new_date, new_session_id, new_turn):
                related_memory_ids = [memory_id]
                text = _ensure_update_markers(text)
            events.append(
                {
                    "event_id": event_id,
                    "timestamp_order": order,
                    "actor_role": str(turn.get("role") or "user"),
                    "trust_level": "trusted",
                    "visibility_scope": "workspace-stable",
                    "event_type": "comment",
                    "text": text,
                    "related_memory_ids": related_memory_ids,
                    "timestamp": date,
                }
            )
            order += 1

    question_event_id = f"e-{question_id}-question"
    events.append(
        {
            "event_id": question_event_id,
            "timestamp_order": order,
            "actor_role": "user",
            "trust_level": "trusted",
            "visibility_scope": "workspace-stable",
            "event_type": "query",
            "text": str(example.get("question") or ""),
            "related_memory_ids": [memory_id],
            "timestamp": str(example.get("question_date") or ""),
        }
    )

    return {
        "scenario_id": f"lme-{question_id}",
        "domain": "longmemeval_knowledge_update",
        "workflow_context": f"LongMemEval knowledge-update question {question_id}",
        "black_box_task": {"prompt": str(example.get("question") or "")},
        "public_input": {
            "initial_memory": [
                {
                    "memory_id": memory_id,
                    "text": str(old_turn.get("content") or ""),
                    "scope": "user-profile",
                    "source_event_ids": [first_event_id or question_event_id],
                }
            ],
            "event_trace": events,
        },
        "metadata": {
            "source": "longmemeval_oracle",
            "question_id": question_id,
            "question_type": example.get("question_type"),
            "gold_answer": example.get("answer"),
        },
    }


def build_ku_supersede_actions_text(view: Any) -> str | None:
    """Deterministic KU policy: supersede stale memory with latest replacement."""
    if not view.candidate_beliefs or not view.candidate_replacement_beliefs:
        return None
    target = view.candidate_beliefs[0].belief_id
    replacement = view.candidate_replacement_beliefs[-1].belief_id
    return json.dumps(
        [
            {
                "action_type": "SUPERSEDES",
                "target_belief_id": target,
                "replacement_belief_id": replacement,
                "evidence_ids": [view.new_evidence.evidence_id],
                "rationale": "LongMemEval knowledge-update: authorize revised memory",
            }
        ],
        ensure_ascii=False,
    )


def authorized_memory_lines(view: Any, response: dict[str, Any]) -> list[str]:
    """Return memory text authorized after DPA projection."""
    memory_state = response.get("memory_state") or {}
    lines: list[str] = []

    if any(status == "outdated" for status in memory_state.values()):
        if view.candidate_replacement_beliefs:
            lines.append(view.candidate_replacement_beliefs[-1].proposition)
            return lines

    for belief in view.candidate_beliefs:
        status = memory_state.get(belief.belief_id, "current")
        if status != "outdated":
            lines.append(belief.proposition)
    for belief in view.candidate_replacement_beliefs:
        lines.append(belief.proposition)

    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        key = re.sub(r"\s+", " ", line.strip())
        if key and key not in seen:
            seen.add(key)
            deduped.append(line.strip())
    return deduped


def build_answer_messages(question: str, view: Any, response: dict[str, Any]) -> list[dict[str, str]]:
    memory_lines = authorized_memory_lines(view, response)
    memory_block = "\n".join(f"- {line}" for line in memory_lines) or "- (no authorized memory)"
    return [
        {
            "role": "system",
            "content": (
                "You answer questions using only the authorized memory below. "
                "Reply with a concise direct answer only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Authorized memory:\n{memory_block}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            ),
        },
    ]

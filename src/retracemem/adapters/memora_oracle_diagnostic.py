from __future__ import annotations

import json
from typing import Any

from retracemem.methods.contracts import SharedCandidateView
from retracemem.schemas import BeliefNode, EvidenceNode


def flatten_values(value: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "value" in value:
            items.append(value)
        else:
            for child in value.values():
                items.extend(flatten_values(child))
    elif isinstance(value, list):
        for child in value:
            items.extend(flatten_values(child))
    return items


def evidence_text_for_sessions(
    sessions: list[dict[str, Any]], session_ids: set[int],
) -> str:
    seen_texts: set[str] = set()
    parts: list[str] = []
    for session in sessions:
        sid = session.get("session_id")
        try:
            sid_int = int(sid)
        except Exception:
            sid_int = -1
        if sid_int not in session_ids:
            continue
        turns: list[str] = []
        for turn in session.get("conversation", []):
            if isinstance(turn, dict):
                speaker = turn.get("speaker") or turn.get("role") or "unknown"
                message = turn.get("message") or turn.get("content") or ""
                if message:
                    turns.append(f"{speaker}: {message}")
        if turns:
            block = f"session {sid}: " + "\n".join(turns)
            if block not in seen_texts:
                seen_texts.add(block)
                parts.append(block)
    return "\n\n".join(parts)


def build_oracle_diagnostic_view(
    period: str, persona: str, question: dict[str, Any],
    sessions: list[dict[str, Any]],
) -> tuple[SharedCandidateView, dict[str, Any]]:
    qid = str(question.get("question_id"))
    memory_items = flatten_values(question.get("memory_evidence", {}))
    forgetting_items = flatten_values(question.get("forgetting_evidence", {}))
    session_ids = {
        int(item.get("session_id"))
        for item in memory_items + forgetting_items
        if str(item.get("session_id", "")).isdigit()
    }
    context_text = evidence_text_for_sessions(sessions, session_ids)
    if not context_text:
        context_text = json.dumps(
            {"memory_evidence": question.get("memory_evidence"),
             "forgetting_evidence": question.get("forgetting_evidence")},
            ensure_ascii=False,
        )
    evidence = EvidenceNode(
        evidence_id=f"memora:{period}:{persona}:{qid}:evidence",
        session_id=f"{period}:{persona}",
        timestamp=question.get("question_date"),
        text=context_text,
        source_dataset="memora",
        source_pointer=f"{period}/{persona}/{qid}",
    )
    beliefs: list[BeliefNode] = []
    for idx, item in enumerate(memory_items):
        val = str(item.get("value") or "").strip()
        if val:
            beliefs.append(BeliefNode(
                belief_id=f"b:{period}:{persona}:{qid}:memory:{idx}",
                proposition=val,
                source_evidence_ids=(evidence.evidence_id,),
                confidence=1.0,
                metadata={"memora_role": "memory_presence",
                          "session_id": item.get("session_id"),
                          "question_date": question.get("question_date")},
            ))
    for idx, item in enumerate(forgetting_items):
        val = str(item.get("value") or "").strip()
        if val:
            beliefs.append(BeliefNode(
                belief_id=f"b:{period}:{persona}:{qid}:forget:{idx}",
                proposition=val,
                source_evidence_ids=(evidence.evidence_id,),
                confidence=1.0,
                metadata={"memora_role": "forgetting_absence",
                          "session_id": item.get("session_id"),
                          "question_date": question.get("question_date")},
            ))
    if not beliefs:
        beliefs.append(BeliefNode(
            belief_id=f"b:{period}:{persona}:{qid}:fallback",
            proposition=str(question.get("question") or ""),
            source_evidence_ids=(evidence.evidence_id,),
            confidence=0.5,
            metadata={"memora_role": "fallback",
                      "question_date": question.get("question_date")},
        ))
    view = SharedCandidateView(
        instance_id=f"memora:{period}:{persona}:{qid}",
        query_id=f"memora:{period}:{persona}:{qid}",
        query=str(question.get("question") or ""),
        evidence_context=(evidence,),
        new_evidence=evidence,
        candidate_beliefs=tuple(beliefs),
        candidate_replacement_beliefs=(),
        candidate_conditions_by_belief=tuple((b.belief_id, ()) for b in beliefs),
        dependency_edges_by_belief=tuple((b.belief_id, ()) for b in beliefs),
    )
    view_meta = {
        "candidate_belief_count": len(beliefs),
        "evidence_chars": len(context_text),
        "selected_sessions": len(session_ids),
    }
    return view, view_meta

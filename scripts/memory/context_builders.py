"""Construct model prompts for external memory baselines on MemPatch-Bench.

All baselines share the same five-field JSON output schema via
:func:`benchmark.model_runner.build_prompt`. They differ only in how
``public_input`` (and optional memory-system fields) are prepared.
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any

from benchmark.general_taxonomy import TASK_TYPES, canonical_hidden_gold_fields
from benchmark.model_runner import build_prompt
from benchmark.public_view import public_scenario_view

BASELINE_IDS: tuple[str, ...] = (
    "structured_direct",
    "full_context",
    "vanilla_rag",
    "bm25_rag",
    "time_aware_rag",
    "summary_memory",
    "mem0",
    "mem0g",
    "a_mem",
    "oracle_evidence",
    "oracle_memory_state",
)

# Paper baselines contain only methods that use public scenario fields.
PAPER_MAIN_BASELINE_IDS: tuple[str, ...] = (
    "structured_direct",
    "full_context",
    "vanilla_rag",
    "time_aware_rag",
    "summary_memory",
)

# Optional supplement-only frozen systems. They are deliberately excluded from
# the main formal baseline table.
PAPER_SUPPLEMENT_BASELINE_IDS: tuple[str, ...] = (
    "mem0",
    "a_mem",
)

PAPER_APPENDIX_BASELINE_IDS: tuple[str, ...] = (
    "bm25_rag",
    "mem0g",
)

# Hidden-gold diagnostic upper bounds. These are never paper baselines.
DIAGNOSTIC_UPPER_BOUND_IDS: tuple[str, ...] = (
    "oracle_evidence",
    "oracle_memory_state",
)

BASELINE_DISPLAY_NAMES: dict[str, str] = {
    "structured_direct": "Frozen Direct Prompting",
    "full_context": "Full Context",
    "vanilla_rag": "Lexical RAG",
    "bm25_rag": "BM25 RAG",
    "time_aware_rag": "Time-Aware RAG",
    "summary_memory": "Summary Memory",
    "mem0": "Mem0-style Proxy",
    "mem0g": "Mem0g-style Proxy",
    "a_mem": "A-MEM-style Proxy",
    "oracle_evidence": "Oracle Evidence (diagnostic)",
    "oracle_memory_state": "Oracle Memory State (diagnostic)",
}

TOKEN_RE = re.compile(r"[a-z0-9_\-]+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def _query_text(view: dict[str, Any]) -> str:
    parts: list[str] = []
    if view.get("workflow_context"):
        parts.append(str(view["workflow_context"]))
    for key in TASK_TYPES:
        task = view.get(key)
        if isinstance(task, dict):
            for field in ("query", "instruction", "prompt", "question"):
                if task.get(field):
                    parts.append(str(task[field]))
    return " ".join(parts)


def _events(public_input: dict[str, Any]) -> list[dict[str, Any]]:
    events = public_input.get("event_trace") or []
    return [ev for ev in events if isinstance(ev, dict)]


def _memories(public_input: dict[str, Any]) -> list[dict[str, Any]]:
    memories = public_input.get("initial_memory") or []
    return [mem for mem in memories if isinstance(mem, dict)]


def _sort_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        events,
        key=lambda ev: (
            ev.get("timestamp_order", 0),
            str(ev.get("timestamp") or ""),
            str(ev.get("event_id") or ""),
        ),
    )


def _overlap_score(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    doc_tokens = set(_tokenize(text))
    if not doc_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / math.sqrt(len(query_tokens))


def _bm25_scores(query_tokens: list[str], docs: list[str], *, k1: float = 1.2, b: float = 0.75) -> list[float]:
    if not query_tokens or not docs:
        return [0.0] * len(docs)
    tokenized = [_tokenize(doc) for doc in docs]
    df: dict[str, int] = {}
    for tokens in tokenized:
        for tok in set(tokens):
            df[tok] = df.get(tok, 0) + 1
    n_docs = len(docs)
    avg_len = sum(len(t) for t in tokenized) / max(n_docs, 1)
    scores: list[float] = []
    for tokens in tokenized:
        doc_len = len(tokens) or 1
        tf_map: dict[str, int] = {}
        for tok in tokens:
            tf_map[tok] = tf_map.get(tok, 0) + 1
        score = 0.0
        for qt in set(query_tokens):
            if qt not in tf_map:
                continue
            idf = math.log(1 + (n_docs - df.get(qt, 0) + 0.5) / (df.get(qt, 0) + 0.5))
            tf = tf_map[qt]
            denom = tf + k1 * (1 - b + b * doc_len / avg_len)
            score += idf * (tf * (k1 + 1)) / denom
        scores.append(score)
    return scores


def _select_events(
    events: list[dict[str, Any]],
    *,
    query: str,
    top_k: int,
    scorer: str,
    recency_weight: float = 0.0,
) -> list[dict[str, Any]]:
    if not events or top_k <= 0:
        return []
    ordered = _sort_events(events)
    if top_k >= len(ordered):
        return ordered

    query_tokens = _tokenize(query)
    texts = [str(ev.get("text") or "") for ev in ordered]
    if scorer == "bm25":
        raw_scores = _bm25_scores(query_tokens, texts)
    else:
        qset = set(query_tokens)
        raw_scores = [_overlap_score(qset, text) for text in texts]

    if recency_weight > 0:
        max_order = max(int(ev.get("timestamp_order") or 0) for ev in ordered) or 1
        boosted: list[float] = []
        for ev, base in zip(ordered, raw_scores):
            order = int(ev.get("timestamp_order") or 0)
            recency = order / max_order
            boosted.append(base + recency_weight * recency)
        raw_scores = boosted

    ranked = sorted(zip(ordered, raw_scores), key=lambda item: item[1], reverse=True)
    top = [ev for ev, _ in ranked[:top_k]]
    return _sort_events(top)


def _summary_from_events(events: list[dict[str, Any]], *, max_chars: int = 2400) -> str:
    lines: list[str] = []
    for ev in _sort_events(events):
        order = ev.get("timestamp_order", "?")
        eid = ev.get("event_id", "?")
        text = re.sub(r"\s+", " ", str(ev.get("text") or "")).strip()
        if len(text) > 160:
            text = text[:157] + "..."
        lines.append(f"- t{order} [{eid}] {text}")
    summary = "\n".join(lines)
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3] + "..."
    return summary


def _mem0_units(public_input: dict[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for mem in _memories(public_input):
        units.append(
            {
                "unit_id": mem.get("memory_id"),
                "kind": "initial_memory",
                "text": mem.get("text"),
                "scope": mem.get("scope"),
                "source_event_ids": list(mem.get("source_event_ids") or []),
            }
        )
    for ev in _sort_events(_events(public_input)):
        for memory_id in ev.get("related_memory_ids") or []:
            units.append(
                {
                    "unit_id": f"{memory_id}::ev::{ev.get('event_id')}",
                    "kind": "event_update",
                    "memory_id": memory_id,
                    "text": ev.get("text"),
                    "event_id": ev.get("event_id"),
                    "timestamp_order": ev.get("timestamp_order"),
                }
            )
    return units


def _retrieve_units(units: list[dict[str, Any]], query: str, *, top_k: int) -> list[dict[str, Any]]:
    qset = set(_tokenize(query))
    scored = [
        (unit, _overlap_score(qset, str(unit.get("text") or "")))
        for unit in units
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [unit for unit, score in scored[:top_k] if score > 0] or [unit for unit, _ in scored[:top_k]]


def _mem0g_graph(units: list[dict[str, Any]], public_input: dict[str, Any]) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for mem in _memories(public_input):
        mid = str(mem.get("memory_id") or "")
        for eid in mem.get("source_event_ids") or []:
            key = (mid, str(eid), "sourced_from")
            if key not in seen:
                seen.add(key)
                edges.append({"from": mid, "to": str(eid), "relation": "sourced_from"})
    for ev in _events(public_input):
        eid = str(ev.get("event_id") or "")
        for mid in ev.get("related_memory_ids") or []:
            key = (str(mid), eid, "related_event")
            if key not in seen:
                seen.add(key)
                edges.append({"from": str(mid), "to": eid, "relation": "related_event"})
    unit_ids = {str(u.get("unit_id")) for u in units}
    return [e for e in edges if e["from"] in unit_ids or e["to"] in unit_ids][:40]


def _a_mem_notes(public_input: dict[str, Any]) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    prev_id: str | None = None
    for ev in _sort_events(_events(public_input)):
        eid = str(ev.get("event_id") or "")
        links = [str(x) for x in (ev.get("related_memory_ids") or [])]
        if prev_id:
            links.append(prev_id)
        notes.append(
            {
                "note_id": eid,
                "text": ev.get("text"),
                "timestamp_order": ev.get("timestamp_order"),
                "links": sorted(set(links)),
            }
        )
        prev_id = eid
    return notes


def _retrieve_a_mem_notes(notes: list[dict[str, Any]], query: str, *, top_k: int) -> list[dict[str, Any]]:
    qset = set(_tokenize(query))
    scored = [(n, _overlap_score(qset, str(n.get("text") or ""))) for n in notes]
    scored.sort(key=lambda item: item[1], reverse=True)
    seeds = [n for n, s in scored[:top_k]]
    chosen: dict[str, dict[str, Any]] = {str(n["note_id"]): n for n in seeds}
    for seed in list(seeds):
        for link in seed.get("links") or []:
            for note in notes:
                if str(note.get("note_id")) == link:
                    chosen[str(note["note_id"])] = note
    return list(chosen.values())


def build_baseline_view(
    scenario: dict[str, Any],
    baseline: str,
    *,
    rag_top_k: int = 8,
) -> dict[str, Any]:
    """Return a model-visible view for ``baseline`` (before ``build_prompt``)."""
    if baseline not in BASELINE_IDS:
        raise ValueError(f"unknown baseline {baseline!r}; expected one of {BASELINE_IDS}")

    view = public_scenario_view(scenario)
    public_input = copy.deepcopy(view.get("public_input") or {})
    events = _events(public_input)
    query = _query_text(view)

    if baseline == "structured_direct":
        pass

    elif baseline == "full_context":
        public_input["event_trace"] = _sort_events(events)
        view["baseline_context"] = {
            "mode": "full_context",
            "instruction": (
                "Read the complete chronological event_trace and initial_memory. "
                "Newer timestamp_order values may supersede older beliefs."
            ),
        }

    elif baseline == "vanilla_rag":
        selected = _select_events(events, query=query, top_k=rag_top_k, scorer="overlap")
        public_input["event_trace"] = selected
        view["baseline_context"] = {
            "mode": "vanilla_rag",
            "retrieved_event_ids": [ev.get("event_id") for ev in selected],
            "instruction": "Answer from retrieved events plus initial_memory.",
        }

    elif baseline == "bm25_rag":
        selected = _select_events(events, query=query, top_k=rag_top_k, scorer="bm25")
        public_input["event_trace"] = selected
        view["baseline_context"] = {
            "mode": "bm25_rag",
            "retrieved_event_ids": [ev.get("event_id") for ev in selected],
        }

    elif baseline == "time_aware_rag":
        selected = _select_events(
            events,
            query=query,
            top_k=rag_top_k,
            scorer="overlap",
            recency_weight=0.35,
        )
        public_input["event_trace"] = _sort_events(selected)
        view["baseline_context"] = {
            "mode": "time_aware_rag",
            "retrieved_event_ids": [ev.get("event_id") for ev in selected],
            "instruction": (
                "Retrieved events are time-sorted. Newer evidence may invalidate older memory."
            ),
        }

    elif baseline == "summary_memory":
        summary = _summary_from_events(events)
        public_input["event_trace"] = []
        view["baseline_context"] = {
            "mode": "summary_memory",
            "rolling_summary": summary,
            "instruction": "Use the rolling summary plus initial_memory; summary may lose detail.",
        }

    elif baseline in {"mem0", "mem0g"}:
        units = _mem0_units(public_input)
        retrieved = _retrieve_units(units, query, top_k=rag_top_k)
        recent = _sort_events(events)[-3:]
        public_input["event_trace"] = recent
        ctx: dict[str, Any] = {
            "mode": baseline,
            "retrieved_memory_units": retrieved,
            "instruction": "Use retrieved Mem0-style units plus recent events.",
        }
        if baseline == "mem0g":
            ctx["memory_graph_edges"] = _mem0g_graph(retrieved, public_input)
        view["baseline_context"] = ctx

    elif baseline == "a_mem":
        notes = _a_mem_notes(public_input)
        retrieved = _retrieve_a_mem_notes(notes, query, top_k=rag_top_k)
        public_input["event_trace"] = _sort_events(events)[-3:]
        view["baseline_context"] = {
            "mode": "a_mem",
            "linked_notes": retrieved,
            "instruction": "Use retrieved linked notes (agentic memory) plus recent events.",
        }

    elif baseline == "oracle_evidence":
        gold = canonical_hidden_gold_fields(scenario.get("hidden_gold") or {})
        view["baseline_context"] = {
            "mode": "oracle_evidence_upper_bound",
            "diagnostic_oracle_evidence_event_ids": gold["expected_evidence_event_ids"],
            "instruction": (
                "Diagnostic upper bound: minimal gold evidence event IDs are provided. "
                "Still output full structured JSON."
            ),
        }

    elif baseline == "oracle_memory_state":
        gold = canonical_hidden_gold_fields(scenario.get("hidden_gold") or {})
        view["baseline_context"] = {
            "mode": "oracle_memory_state_upper_bound",
            "diagnostic_oracle_memory_state": gold["expected_memory_state"],
            "instruction": (
                "Diagnostic upper bound: gold memory_state labels are provided for each memory_id."
            ),
        }

    view["public_input"] = public_input
    return view


def build_baseline_prompt(
    scenario: dict[str, Any],
    baseline: str,
    *,
    rag_top_k: int = 8,
) -> str:
    """Build strict-JSON user prompt for a baseline variant."""
    view = build_baseline_view(scenario, baseline, rag_top_k=rag_top_k)
    return build_prompt(view)

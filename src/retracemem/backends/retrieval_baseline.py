from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class _Evidence:
    id: str
    text: str
    source_id: str
    session_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrievalBaselineBackend:
    """Deterministic raw-text retrieval baseline.

    This backend intentionally does no memory rewriting or generation. It stores
    raw session/evidence text and ranks it with a simple lexical score so
    benchmark harnesses have a no-dependency baseline.
    """

    method = "retrieval_baseline"

    def __init__(self) -> None:
        self._evidence_by_user: dict[str, list[_Evidence]] = {}

    def reset_user(self, user_id: str) -> None:
        self._evidence_by_user[user_id] = []

    def ingest_session(
        self,
        user_id: str,
        session: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._ensure_user(user_id)
        session_metadata = dict(metadata or {})
        session_id = str(
            session.get("session_id")
            or session.get("id")
            or session_metadata.get("session_id")
            or f"session_{len(self._evidence_by_user[user_id]) + 1}"
        )
        source_id = str(session.get("source_id") or session_metadata.get("source_id") or session_id)

        for text, chunk_metadata in self._extract_text_chunks(session):
            if not text.strip():
                continue
            evidence_index = len(self._evidence_by_user[user_id]) + 1
            evidence_metadata = {**session_metadata, **chunk_metadata}
            evidence_id = str(
                evidence_metadata.get("evidence_id")
                or evidence_metadata.get("id")
                or f"{session_id}_evidence_{evidence_index:04d}"
            )
            self._evidence_by_user[user_id].append(
                _Evidence(
                    id=evidence_id,
                    text=text,
                    source_id=source_id,
                    session_id=session_id,
                    metadata=evidence_metadata,
                )
            )

    def search(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        del metadata
        self._ensure_user(user_id)
        if limit <= 0:
            return []

        query_tokens = _tokens(query)
        if not query_tokens and not query.strip():
            return []

        scored: list[tuple[float, int, _Evidence, list[str]]] = []
        normalized_query = _normalize_text(query)
        for index, evidence in enumerate(self._evidence_by_user[user_id]):
            score, match_terms = self._score(query_tokens, normalized_query, evidence.text)
            if score <= 0:
                continue
            scored.append((score, index, evidence, match_terms))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                "id": evidence.id,
                "text": evidence.text,
                "score": score,
                "source_id": evidence.source_id,
                "session_id": evidence.session_id,
                "metadata": dict(evidence.metadata),
                "match_terms": match_terms,
            }
            for score, _index, evidence, match_terms in scored[:limit]
        ]

    def answer(
        self,
        user_id: str,
        query: str,
        retrieved: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        del user_id, metadata
        evidence_ids = ", ".join(str(item.get("id", "")) for item in retrieved) or "none"
        context_lines = [
            f"{index}. [{item.get('id', '')}] {item.get('text', '')}"
            for index, item in enumerate(retrieved, start=1)
        ]
        context = "\n".join(context_lines) if context_lines else "(no retrieved evidence)"
        return (
            "method: retrieval_baseline\n"
            f"query: {query}\n"
            f"retrieved_evidence_ids: {evidence_ids}\n"
            "answer: deterministic retrieval baseline; inspect retrieved evidence.\n"
            f"retrieved_evidence:\n{context}"
        )

    def _ensure_user(self, user_id: str) -> None:
        if user_id not in self._evidence_by_user:
            self.reset_user(user_id)

    def _extract_text_chunks(self, session: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        evidence_items = session.get("evidence")
        if isinstance(evidence_items, list):
            chunks: list[tuple[str, dict[str, Any]]] = []
            for index, item in enumerate(evidence_items, start=1):
                text = _text_from_value(item)
                if text:
                    chunks.append((text, _metadata_from_item(item, index)))
            if chunks:
                return chunks

        for key in ("messages", "turns", "utterances"):
            items = session.get(key)
            if not isinstance(items, list):
                continue
            chunks = []
            for index, item in enumerate(items, start=1):
                text = _text_from_value(item)
                if text:
                    chunks.append((text, _metadata_from_item(item, index)))
            if chunks:
                return chunks

        text = _text_from_value(session.get("text"))
        if text:
            return [(text, {})]

        return [(json.dumps(session, ensure_ascii=False, sort_keys=True), {})] if session else []

    def _score(
        self,
        query_tokens: set[str],
        normalized_query: str,
        text: str,
    ) -> tuple[float, list[str]]:
        normalized_text = _normalize_text(text)
        text_tokens = _tokens(text)
        match_terms = sorted(query_tokens & text_tokens)

        score = 0.0
        if query_tokens:
            score += len(match_terms) / len(query_tokens)
            score += sum(normalized_text.count(term) for term in query_tokens) * 0.01
        if normalized_query and normalized_query in normalized_text:
            score += 2.0
        return round(score, 6), match_terms


def _normalize_text(value: str) -> str:
    return " ".join(_TOKEN_RE.findall(value.lower()))


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.lower()))


def _text_from_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""
    for key in ("text", "content", "message", "utterance", "value"):
        text = value.get(key)
        if isinstance(text, str):
            role = value.get("role") or value.get("speaker")
            return f"{role}: {text}" if isinstance(role, str) and role else text
    return ""


def _metadata_from_item(item: Any, index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"chunk_index": index}
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return {
        **metadata,
        "chunk_index": index,
        **{key: item[key] for key in ("id", "evidence_id", "timestamp", "role", "speaker") if key in item},
    }

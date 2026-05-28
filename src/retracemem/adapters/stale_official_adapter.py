"""Strict non-leaking adapter for the official frozen STALE benchmark dataset.

Loads ``T1_T2_400_FULL.json`` from
``data_external/stale_official_frozen/`` and exposes a clean separation:

- :class:`StaleMethodVisibleScenario` carries only what the method may see:
  ``uid``, ordered ``haystack_sessions`` aligned with ``timestamps``, and the
  three ``probing_queries``.
- :class:`StaleEvaluatorOnlyMetadata` carries gold/auditor fields:
  ``M_old``, ``M_new``, ``explanation``, ``relevant_session_index`` and
  ``type``. These must not be supplied to extraction, retrieval, edge
  prediction, authorization, or final answer generation.

The adapter never injects evaluator-only fields into the method-visible view.
It also writes no output under ``reference/``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_REQUIRED_TOP_KEYS = (
    "uid",
    "M_old",
    "M_new",
    "explanation",
    "probing_queries",
    "relevant_session_index",
    "timestamps",
    "haystack_session",
    "type",
)
_PROBING_KEYS = ("dim1_query", "dim2_query", "dim3_query")
_VALID_TYPES = ("T1", "T2")


@dataclass(frozen=True)
class StaleMethodVisibleScenario:
    uid: str
    haystack_sessions: tuple[tuple[str, ...], ...]
    timestamps: tuple[str, ...]
    probing_queries: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class StaleEvaluatorOnlyMetadata:
    uid: str
    type: str
    m_old: str
    m_new: str
    explanation: str
    relevant_session_index: Any


@dataclass(frozen=True)
class StaleOfficialRecord:
    method_visible: StaleMethodVisibleScenario
    evaluator_only: StaleEvaluatorOnlyMetadata


class StaleOfficialAdapter:
    """Strict loader for the official frozen STALE benchmark dataset."""

    def __init__(
        self,
        dataset_path: str | Path = "data_external/stale_official_frozen/T1_T2_400_FULL.json",
    ) -> None:
        self.dataset_path = Path(dataset_path)

    def exists(self) -> bool:
        return self.dataset_path.is_file()

    def load(self) -> tuple[StaleOfficialRecord, ...]:
        if not self.exists():
            raise FileNotFoundError(
                f"Official frozen STALE dataset not found at {self.dataset_path}. "
                "Place T1_T2_400_FULL.json under data_external/stale_official_frozen/."
            )
        payload = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(
                "Official frozen STALE dataset must be a JSON list of records."
            )
        return tuple(parse_record(item, index=index) for index, item in enumerate(payload))

    def load_method_visible(self) -> tuple[StaleMethodVisibleScenario, ...]:
        return tuple(record.method_visible for record in self.load())

    def stratify_by_type(
        self, records: tuple[StaleOfficialRecord, ...]
    ) -> dict[str, tuple[StaleOfficialRecord, ...]]:
        buckets: dict[str, list[StaleOfficialRecord]] = {key: [] for key in _VALID_TYPES}
        for record in records:
            buckets.setdefault(record.evaluator_only.type, []).append(record)
        return {key: tuple(value) for key, value in buckets.items()}


def parse_record(item: Any, *, index: int) -> StaleOfficialRecord:
    if not isinstance(item, dict):
        raise ValueError(f"Record {index} is not a JSON object.")
    missing = [key for key in _REQUIRED_TOP_KEYS if key not in item]
    if missing:
        raise ValueError(f"Record {index} missing required fields: {missing}.")

    uid = item["uid"]
    if not isinstance(uid, str) or not uid:
        raise ValueError(f"Record {index} has invalid uid.")

    rtype = item["type"]
    if rtype not in _VALID_TYPES:
        raise ValueError(
            f"Record {index} ({uid}) has unsupported type {rtype!r}; expected one of {_VALID_TYPES}."
        )

    haystack_raw = item["haystack_session"]
    timestamps_raw = item["timestamps"]
    if not isinstance(haystack_raw, list) or not isinstance(timestamps_raw, list):
        raise ValueError(f"Record {index} ({uid}) has invalid haystack_session or timestamps.")
    if len(haystack_raw) != len(timestamps_raw):
        raise ValueError(
            f"Record {index} ({uid}) has misaligned haystack_session ({len(haystack_raw)}) "
            f"and timestamps ({len(timestamps_raw)})."
        )
    sessions = tuple(_normalize_session(session, index, uid) for session in haystack_raw)
    timestamps = tuple(_normalize_timestamp(ts, index, uid) for ts in timestamps_raw)

    probing_raw = item["probing_queries"]
    if not isinstance(probing_raw, dict):
        raise ValueError(f"Record {index} ({uid}) has invalid probing_queries.")
    probing: list[tuple[str, str]] = []
    for key in _PROBING_KEYS:
        value = probing_raw.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Record {index} ({uid}) is missing probing query {key!r}."
            )
        probing.append((key, value))

    method_visible = StaleMethodVisibleScenario(
        uid=uid,
        haystack_sessions=sessions,
        timestamps=timestamps,
        probing_queries=tuple(probing),
    )
    evaluator_only = StaleEvaluatorOnlyMetadata(
        uid=uid,
        type=rtype,
        m_old=str(item.get("M_old") or ""),
        m_new=str(item.get("M_new") or ""),
        explanation=str(item.get("explanation") or ""),
        relevant_session_index=item.get("relevant_session_index"),
    )
    return StaleOfficialRecord(method_visible=method_visible, evaluator_only=evaluator_only)


def _normalize_session(session: Any, index: int, uid: str) -> tuple[str, ...]:
    if isinstance(session, list):
        return tuple(_coerce_turn(turn) for turn in session)
    if isinstance(session, str):
        return (session,)
    raise ValueError(
        f"Record {index} ({uid}) has unsupported haystack_session entry type {type(session).__name__}."
    )


def _coerce_turn(turn: Any) -> str:
    if isinstance(turn, str):
        return turn
    if isinstance(turn, list):
        return "\n".join(_coerce_turn(part) for part in turn)
    return str(turn)


def _normalize_timestamp(ts: Any, index: int, uid: str) -> str:
    if not isinstance(ts, str) or not ts:
        raise ValueError(f"Record {index} ({uid}) has invalid timestamp entry.")
    return ts

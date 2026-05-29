from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from retracemem.schemas import EvidenceNode


@dataclass(frozen=True)
class SessionRecord:
    role: str
    content: str


@dataclass(frozen=True)
class StaleWriteHistory:
    uid: str
    sessions: tuple[tuple[SessionRecord, ...], ...]
    timestamps: tuple[str, ...]


@dataclass(frozen=True)
class StaleProbeTask:
    uid: str
    dimension: str
    query: str
    memory_snapshot_id: str | None = None


@dataclass(frozen=True)
class StaleGoldRecord:
    uid: str
    m_old: str
    m_new: str
    explanation: str
    relevant_session_index: tuple[int, ...]
    conflict_type: str


def assert_no_evaluation_leakage(payload: Any) -> None:
    """Recursively checks and raises ValueError if any blacklisted fields or values are found."""
    blacklist_keys = {
        "M_old", "M_new", "explanation", "relevant_session_index", "type",
        "m_old", "m_new", "conflict_type", "relevant_session_index", "relevant_session_indices"
    }
    
    def _check(val: Any) -> None:
        if isinstance(val, dict):
            for k, v in val.items():
                if k in blacklist_keys:
                    raise ValueError(f"Evaluation leakage detected: key '{k}' found in payload.")
                _check(v)
        elif isinstance(val, (list, tuple)):
            for item in val:
                _check(item)
        elif hasattr(val, "__dict__"):
            for k, v in val.__dict__.items():
                if k in blacklist_keys:
                    raise ValueError(f"Evaluation leakage detected: attribute '{k}' found in object.")
                _check(v)

    _check(payload)


def split_stale_record(
    raw: dict[str, Any],
) -> tuple[StaleWriteHistory, tuple[StaleProbeTask, ...], StaleGoldRecord]:
    """Splits raw STALE dataset record into write history, probe tasks, and gold record."""
    uid = str(raw["uid"])
    
    # Construct write history
    sessions_list = []
    for s in raw["haystack_session"]:
        session_turns = tuple(SessionRecord(role=str(turn["role"]), content=str(turn["content"])) for turn in s)
        sessions_list.append(session_turns)
    write_history = StaleWriteHistory(
        uid=uid,
        sessions=tuple(sessions_list),
        timestamps=tuple(str(t) for t in raw["timestamps"]),
    )
    assert_no_evaluation_leakage(write_history)
    
    # Construct probe tasks
    probes = []
    for dim, query_text in sorted(raw["probing_queries"].items()):
        probes.append(
            StaleProbeTask(
                uid=uid,
                dimension=dim,
                query=query_text,
            )
        )
    assert_no_evaluation_leakage(probes)
    
    # Construct gold record
    gold = StaleGoldRecord(
        uid=uid,
        m_old=str(raw["M_old"]),
        m_new=str(raw["M_new"]),
        explanation=str(raw["explanation"]),
        relevant_session_index=tuple(int(idx) for idx in raw["relevant_session_index"]),
        conflict_type=str(raw["type"]),
    )
    
    return write_history, tuple(probes), gold


def iter_chronological_sessions(
    history: StaleWriteHistory,
) -> tuple[EvidenceNode, ...]:
    """Generates chronological EvidenceNode objects from StaleWriteHistory."""
    nodes = []
    for idx, (session_turns, timestamp) in enumerate(zip(history.sessions, history.timestamps)):
        session_id = f"s_{idx}"
        evidence_id = f"e_{idx}"
        
        parts = []
        for turn in session_turns:
            parts.append(f"{turn.role}: {turn.content}")
        text = "\n".join(parts)
        
        nodes.append(
            EvidenceNode(
                evidence_id=evidence_id,
                session_id=session_id,
                timestamp=timestamp,
                text=text,
                source_dataset="stale",
                source_pointer=f"stale://{history.uid}/{idx}",
                is_raw_source=True,
            )
        )
    return tuple(nodes)

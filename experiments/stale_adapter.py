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


@dataclass(frozen=True)
class CupMemRevisionCandidate:
    old_state_id: str
    old_state_text: str
    old_source_evidence: tuple[str, ...]
    new_evidence_id: str
    new_evidence_text: str
    candidate_replacement_text: str | None
    affected_region: str | None
    upstream_trace: dict[str, Any]


def map_cupmem_candidate_to_retrace_view(
    candidate: CupMemRevisionCandidate,
    *,
    pre_induced_requirements: tuple[DependencyEdge, ...],
) -> SharedCandidateView:
    """Maps CUPMem candidate to ReTrace view under no-leakage constraints."""
    from retracemem.schemas import BeliefNode, ConditionNode, DependencyEdge
    from retracemem.methods.contracts import SharedCandidateView

    new_ev = EvidenceNode(
        evidence_id=candidate.new_evidence_id,
        session_id="s_current",
        timestamp="2026-05-29T00:00:00Z",
        text=candidate.new_evidence_text,
        source_dataset="cupmem_adapter",
        source_pointer=f"cupmem://{candidate.new_evidence_id}"
    )
    
    old_belief = BeliefNode(
        belief_id=candidate.old_state_id,
        proposition=candidate.old_state_text,
        source_evidence_ids=candidate.old_source_evidence
    )
    
    replacement_beliefs = ()
    if candidate.candidate_replacement_text:
        replacement_beliefs = (
            BeliefNode(
                belief_id=f"rep_{candidate.old_state_id}",
                proposition=candidate.candidate_replacement_text,
                source_evidence_ids=(candidate.new_evidence_id,)
            ),
        )
        
    conds = []
    deps = []
    for r in pre_induced_requirements:
        if r.belief_id == candidate.old_state_id:
            deps.append(r)
            conds.append(ConditionNode(condition_id=r.condition_id, scope_id="user_scope", text=r.condition_id))
            
    return SharedCandidateView(
        instance_id=candidate.upstream_trace.get("instance_id", "cupmem_inst"),
        query_id=candidate.upstream_trace.get("query_id", "cupmem_q"),
        query=candidate.upstream_trace.get("query", ""),
        evidence_context=(new_ev,),
        new_evidence=new_ev,
        candidate_beliefs=(old_belief,),
        candidate_replacement_beliefs=replacement_beliefs,
        candidate_conditions_by_belief=((candidate.old_state_id, tuple(conds)),) if conds else (),
        dependency_edges_by_belief=((candidate.old_state_id, tuple(deps)),) if deps else ()
    )


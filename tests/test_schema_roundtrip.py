from __future__ import annotations
import dataclasses
import json
from typing import Any
from retracemem.schemas import (
    EvidenceRecord,
    BeliefRecord,
    ConditionRecord,
    RelationPredictionRecord,
    RelationLabel,
    AuthorizationRecord,
    AuthorizationStatus,
    DefeatPathRecord,
    DefeatPathType,
    QueryRecord,
    MethodTraceRecord,
    ScoreRecord,
    RunManifest,
    ModelCallTrace,
)


def _roundtrip(obj: Any, cls: type) -> Any:
    # Serialize to JSON string
    serialized_str = json.dumps(dataclasses.asdict(obj))
    # Deserialize to dict
    data = json.loads(serialized_str)
    # Reconstruct the dataclass
    fields = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in fields}
    return cls(**kwargs)


def test_evidence_record_roundtrip() -> None:
    record = EvidenceRecord(
        evidence_id="ev_001",
        session_id="sess_001",
        timestamp="2026-05-27T00:00:00Z",
        text="User broke their leg.",
        source_dataset="manual_audit",
        source_pointer="session_1_line_2",
        is_raw_source=True,
        metadata={"tags": ["health"]},
    )
    restored = _roundtrip(record, EvidenceRecord)
    assert restored.evidence_id == record.evidence_id
    assert restored.text == record.text
    assert restored.metadata == record.metadata


def test_belief_record_roundtrip() -> None:
    record = BeliefRecord(
        belief_id="bel_001",
        proposition="User commutes by bicycle.",
        source_evidence_ids=["ev_001"],
        source_span="commutes by bicycle",
        timestamp="2026-05-27T00:00:00Z",
        extractor_version="v1.0",
        confidence=0.95,
        metadata={"priority": "high"},
    )
    restored = _roundtrip(record, BeliefRecord)
    assert restored.belief_id == record.belief_id
    assert restored.proposition == record.proposition
    assert restored.source_evidence_ids == record.source_evidence_ids
    assert restored.confidence == record.confidence


def test_condition_record_roundtrip() -> None:
    record = ConditionRecord(
        belief_id="bel_001",
        condition="cycling ability",
        metadata={"reason": "prerequisite"},
    )
    restored = _roundtrip(record, ConditionRecord)
    assert restored.belief_id == record.belief_id
    assert restored.condition == record.condition


def test_relation_prediction_record_roundtrip() -> None:
    record = RelationPredictionRecord(
        relation=RelationLabel.BLOCK,
        evidence_id="ev_002",
        belief_id="bel_001",
        condition="cycling ability",
        rationale="leg is broken",
        confidence=0.99,
    )
    restored = _roundtrip(record, RelationPredictionRecord)
    assert restored.relation == record.relation
    assert restored.condition == record.condition
    assert restored.confidence == record.confidence


def test_authorization_record_roundtrip() -> None:
    record = AuthorizationRecord(
        belief_id="bel_001",
        authorization_status=AuthorizationStatus.BLOCKED,
        reason="leg is broken",
        justification_path=["ev_002"],
    )
    restored = _roundtrip(record, AuthorizationRecord)
    assert restored.belief_id == record.belief_id
    assert restored.authorization_status == record.authorization_status
    assert restored.justification_path == record.justification_path


def test_defeat_path_record_roundtrip() -> None:
    record = DefeatPathRecord(
        path_id="path_001",
        path_type=DefeatPathType.PREREQUISITE_BLOCK,
        source_belief_id="bel_001",
        evidence_id="ev_002",
    )
    restored = _roundtrip(record, DefeatPathRecord)
    assert restored.path_id == record.path_id
    assert restored.path_type == record.path_type
    assert restored.source_belief_id == record.source_belief_id


def test_query_record_roundtrip() -> None:
    record = QueryRecord(
        query_id="q_001",
        query_text="How should they commute?",
        timestamp="2026-05-27T01:00:00Z",
    )
    restored = _roundtrip(record, QueryRecord)
    assert restored.query_id == record.query_id
    assert restored.query_text == record.query_text


def test_method_trace_record_roundtrip() -> None:
    record = MethodTraceRecord(
        example_id="ex_001",
        method_name="retrace_heuristic",
        upstream_commit="abc1234",
        query="test query",
        candidate_evidence_ids=["ev_001"],
        decision_payload={"authorized": False},
        answer="cannot ride a bike",
        model_config_id="gpt-4o",
        token_counts={"prompt": 100},
        call_counts={"verify": 1},
        errors=["timeout"],
    )
    restored = _roundtrip(record, MethodTraceRecord)
    assert restored.example_id == record.example_id
    assert restored.method_name == record.method_name
    assert restored.token_counts == record.token_counts
    assert restored.errors == record.errors


def test_score_record_roundtrip() -> None:
    record = ScoreRecord(
        example_id="ex_001",
        benchmark="stale",
        method_name="retrace_heuristic",
        official_scores={"accuracy": 1.0},
        local_diagnostics={"revision_correct": True},
        evaluator_version="v2",
        run_manifest_id="manifest_001",
    )
    restored = _roundtrip(record, ScoreRecord)
    assert restored.example_id == record.example_id
    assert restored.benchmark == record.benchmark
    assert restored.official_scores == record.official_scores


def test_run_manifest_roundtrip() -> None:
    record = RunManifest(
        run_manifest_id="manifest_001",
        method_name="retrace_heuristic",
        model_config_id="gpt-4o",
        timestamp="2026-05-27T02:00:00Z",
        upstream_commit="abc1234",
        output_path="/tmp/output.jsonl",
        checksum="sha256checksum",
    )
    restored = _roundtrip(record, RunManifest)
    assert restored.run_manifest_id == record.run_manifest_id
    assert restored.output_path == record.output_path
    assert restored.checksum == record.checksum


def test_model_call_trace_roundtrip() -> None:
    record = ModelCallTrace(
        call_id="call_001",
        provider="openai",
        model_id="gpt-4o",
        model_revision_or_api_version="2024-05-13",
        prompt_template_hash="template_hash",
        response_schema_version="v1",
        parser_version="v1",
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
        seed=42,
        input_hash="input_hash",
        condition_context_hash="cond_hash",
        temporal_context_hash="temp_hash",
        status="success",
        response="hello",
        latency_ms=120.5,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        retries=1,
        eligible_for_replay=True,
    )
    restored = _roundtrip(record, ModelCallTrace)
    assert restored.call_id == record.call_id
    assert restored.temperature == record.temperature
    assert restored.eligible_for_replay == record.eligible_for_replay

from __future__ import annotations
import dataclasses
import json
from typing import Any
from retracemem.schemas import (
    AuthorizationRecord,
    AuthorizationStatus,
    AuthorizationTrace,
    BeliefNode,
    BeliefRecord,
    ConditionNode,
    ConditionRecord,
    DefeatPath,
    DefeatPathRecord,
    DefeatPathType,
    DependencyEdge,
    EvidenceEdge,
    EvidenceEdgeType,
    EvidenceNode,
    EvidenceRecord,
    MethodTraceRecord,
    ModelCallTrace,
    QueryRecord,
    RelationLabel,
    RelationPredictionRecord,
    RunManifest,
    ScoreRecord,
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


def _roundtrip_with_tuples(obj: Any, cls: type, tuple_fields: tuple[str, ...] = ()) -> Any:
    """Roundtrip a frozen dataclass that uses ``tuple`` fields.

    `dataclasses.asdict` converts tuples to lists when serialising via JSON;
    the reverse construction needs to coerce them back so the equality checks
    on the original tuples succeed.
    """
    serialized_str = json.dumps(dataclasses.asdict(obj))
    data = json.loads(serialized_str)
    fields = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in fields}
    for name in tuple_fields:
        if name in kwargs and isinstance(kwargs[name], list):
            kwargs[name] = tuple(kwargs[name])
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


# ---------------------------------------------------------------------------
# Canonical Typed Graph Schemas (Refactor Wave 0).
#
# Round-trip and structural-invariant tests for the new schemas introduced
# by `docs/refactor_plan_defeat_path.md` amendments A1-A10. These are the
# only schemas that Wave 1+ runtime code is allowed to import.
# ---------------------------------------------------------------------------


def test_evidence_node_roundtrip() -> None:
    node = EvidenceNode(
        evidence_id="ev_n_001",
        session_id="sess_001",
        timestamp="2026-05-27T00:00:00Z",
        text="User broke their leg.",
        source_dataset="manual_audit",
        source_pointer="session_1_line_2",
        is_raw_source=True,
        metadata={"tags": ["health"]},
    )
    restored = _roundtrip(node, EvidenceNode)
    assert restored == node


def test_belief_node_roundtrip_preserves_tuple_provenance() -> None:
    node = BeliefNode(
        belief_id="bel_n_001",
        proposition="The user usually commutes by bicycle.",
        source_evidence_ids=("ev_n_001", "ev_n_002"),
        source_span="commutes by bicycle",
        extractor_version="manual_fixture_v0",
        confidence=0.9,
        metadata={"topic": "commute"},
    )
    restored = _roundtrip_with_tuples(
        node, BeliefNode, tuple_fields=("source_evidence_ids",)
    )
    assert restored == node
    assert isinstance(restored.source_evidence_ids, tuple)


def test_condition_node_requires_scope_id() -> None:
    """Amendment A7: condition identity is namespaced by scope_id."""

    node_user_a = ConditionNode(
        condition_id="cond_cycling",
        scope_id="user_a",
        text="cycling ability",
    )
    node_user_b = ConditionNode(
        condition_id="cond_cycling",
        scope_id="user_b",
        text="cycling ability",
    )
    assert node_user_a != node_user_b, "scope_id must keep conditions isolated across scopes"
    restored = _roundtrip(node_user_a, ConditionNode)
    assert restored == node_user_a


def test_dependency_edge_carries_provenance() -> None:
    """Amendment A8: provenance fields are first-class, not metadata."""

    edge = DependencyEdge(
        edge_id="dep_edge_001",
        belief_id="bel_n_001",
        condition_id="cond_cycling",
        inducer="manual_fixture",
        edge_type="REQUIRES",
        supporting_evidence_ids=("ev_n_001",),
        model_call_trace_id="call_001",
        confidence=0.8,
        rationale="Bicycle commute requires the user to be physically able to cycle.",
        metadata={"notes": "from boundary audit dev seed"},
    )
    restored = _roundtrip_with_tuples(
        edge, DependencyEdge, tuple_fields=("supporting_evidence_ids",)
    )
    assert restored == edge
    assert restored.edge_type == "REQUIRES"
    assert restored.inducer == "manual_fixture"
    assert restored.model_call_trace_id == "call_001"


def test_evidence_edge_blocks_targets_condition_roundtrip() -> None:
    edge = EvidenceEdge(
        edge_id="ev_edge_blocks_001",
        edge_type=EvidenceEdgeType.BLOCKS,
        evidence_id="ev_n_002",
        target_kind="condition",
        target_id="cond_cycling",
        verifier="heuristic",
        confidence=0.9,
        rationale="Broken leg blocks cycling ability.",
        span="broke their leg",
    )
    restored = _roundtrip(edge, EvidenceEdge)
    assert restored.edge_type == EvidenceEdgeType.BLOCKS
    assert restored.target_kind == "condition"
    assert restored.target_id == "cond_cycling"
    assert restored.replacement_belief_id is None


def test_evidence_edge_supersedes_carries_replacement_belief_id() -> None:
    """Amendment A1: SUPERSEDES preserves the replacement belief id."""

    edge = EvidenceEdge(
        edge_id="ev_edge_super_001",
        edge_type=EvidenceEdgeType.SUPERSEDES,
        evidence_id="ev_n_003",
        target_kind="belief",
        target_id="bel_old_address",
        verifier="prompt",
        replacement_belief_id="bel_new_address",
        rationale="Moved to Cedar Ave.",
    )
    restored = _roundtrip(edge, EvidenceEdge)
    assert restored.edge_type == EvidenceEdgeType.SUPERSEDES
    assert restored.replacement_belief_id == "bel_new_address"
    assert restored.target_id == "bel_old_address"


def test_evidence_edge_reaffirms_targets_belief() -> None:
    """Amendment A3: REAFFIRMS exists and targets a belief."""

    edge = EvidenceEdge(
        edge_id="ev_edge_reaff_001",
        edge_type=EvidenceEdgeType.REAFFIRMS,
        evidence_id="ev_n_004",
        target_kind="belief",
        target_id="bel_lives_in_shanghai",
        verifier="prompt",
        rationale="User confirmed they still live in Shanghai.",
    )
    restored = _roundtrip(edge, EvidenceEdge)
    assert restored.edge_type == EvidenceEdgeType.REAFFIRMS
    assert restored.target_kind == "belief"


def test_defeat_path_direct_supersede_carries_replacement() -> None:
    """Amendment A1: DefeatPath surfaces the replacement belief id."""

    path = DefeatPath(
        path_id="path_super_001",
        path_type=DefeatPathType.DIRECT_SUPERSEDE,
        target_belief_id="bel_old_address",
        supporting_dependency_edge_ids=(),
        supporting_evidence_edge_ids=("ev_edge_super_001",),
        replacement_belief_id="bel_new_address",
        as_of_evidence_id="ev_n_003",
    )
    restored = _roundtrip_with_tuples(
        path,
        DefeatPath,
        tuple_fields=(
            "supporting_dependency_edge_ids",
            "supporting_evidence_edge_ids",
        ),
    )
    assert restored == path
    assert restored.replacement_belief_id == "bel_new_address"


def test_defeat_path_prerequisite_block_carries_two_edge_kinds() -> None:
    path = DefeatPath(
        path_id="path_block_001",
        path_type=DefeatPathType.PREREQUISITE_BLOCK,
        target_belief_id="bel_n_001",
        supporting_dependency_edge_ids=("dep_edge_001",),
        supporting_evidence_edge_ids=("ev_edge_blocks_001",),
        as_of_evidence_id="ev_n_002",
    )
    restored = _roundtrip_with_tuples(
        path,
        DefeatPath,
        tuple_fields=(
            "supporting_dependency_edge_ids",
            "supporting_evidence_edge_ids",
        ),
    )
    assert restored == path
    assert restored.replacement_belief_id is None


def test_authorization_trace_authorized_has_no_defeat_path() -> None:
    """Amendment A5: AUTHORIZED traces have accepted_defeat_path = None."""

    trace = AuthorizationTrace(
        trace_id="trace_auth_001",
        belief_id="bel_food_preference",
        status=AuthorizationStatus.AUTHORIZED,
        accepted_defeat_path=None,
        supporting_evidence_ids=("ev_n_010",),
        as_of_evidence_id="ev_n_010",
    )
    restored = _roundtrip_with_tuples(
        trace, AuthorizationTrace, tuple_fields=("supporting_evidence_ids",)
    )
    assert restored.status == AuthorizationStatus.AUTHORIZED
    assert restored.accepted_defeat_path is None
    assert restored.supporting_evidence_ids == ("ev_n_010",)


def test_authorization_trace_blocked_carries_full_defeat_path() -> None:
    path = DefeatPath(
        path_id="path_block_002",
        path_type=DefeatPathType.PREREQUISITE_BLOCK,
        target_belief_id="bel_n_001",
        supporting_dependency_edge_ids=("dep_edge_001",),
        supporting_evidence_edge_ids=("ev_edge_blocks_001",),
        as_of_evidence_id="ev_n_002",
    )
    trace = AuthorizationTrace(
        trace_id="trace_blocked_001",
        belief_id="bel_n_001",
        status=AuthorizationStatus.BLOCKED,
        accepted_defeat_path=path,
        considered_defeat_paths=(path,),
        as_of_evidence_id="ev_n_002",
    )
    serialized = json.dumps(dataclasses.asdict(trace))
    data = json.loads(serialized)
    # Dataclass round-trip with nested objects requires manual reconstruction.
    accepted = data["accepted_defeat_path"]
    accepted["supporting_dependency_edge_ids"] = tuple(
        accepted["supporting_dependency_edge_ids"]
    )
    accepted["supporting_evidence_edge_ids"] = tuple(
        accepted["supporting_evidence_edge_ids"]
    )
    accepted["path_type"] = DefeatPathType(accepted["path_type"])
    rebuilt_path = DefeatPath(**accepted)
    rebuilt_trace = AuthorizationTrace(
        trace_id=data["trace_id"],
        belief_id=data["belief_id"],
        status=AuthorizationStatus(data["status"]),
        accepted_defeat_path=rebuilt_path,
        considered_defeat_paths=(rebuilt_path,),
        supporting_evidence_ids=tuple(data["supporting_evidence_ids"]),
        as_of_evidence_id=data["as_of_evidence_id"],
    )
    assert rebuilt_trace.status == AuthorizationStatus.BLOCKED
    assert rebuilt_trace.accepted_defeat_path == path


def test_authorization_status_drops_legacy_conditional() -> None:
    """Amendment A2: CONDITIONAL is removed from the canonical status set."""

    assert "CONDITIONAL" not in {status.value for status in AuthorizationStatus}


def test_defeat_path_type_drops_legacy_rollback_release() -> None:
    """ROLLBACK_RELEASE is not a defeat outcome under amendment A2."""

    values = {path_type.value for path_type in DefeatPathType}
    assert "ROLLBACK_RELEASE" not in values
    assert {"DIRECT_SUPERSEDE", "PREREQUISITE_BLOCK", "UNRESOLVED_UNCERTAIN"} <= values


def test_evidence_edge_type_includes_reaffirms() -> None:
    """Amendment A3: REAFFIRMS is part of the canonical evidence edge set."""

    values = {edge_type.value for edge_type in EvidenceEdgeType}
    assert {"BLOCKS", "RELEASES", "SUPERSEDES", "REAFFIRMS", "UNCERTAIN"} == values

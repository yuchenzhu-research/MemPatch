from __future__ import annotations
import dataclasses
import json
from typing import Any
from retracemem.schemas import (
    AuthorizationStatus,
    AuthorizationTrace,
    BeliefNode,
    ConditionNode,
    DefeatPath,
    DefeatPathType,
    DependencyEdge,
    EvidenceEdge,
    EvidenceEdgeType,
    EvidenceNode,
)


def _roundtrip(obj: Any, cls: type) -> Any:
    serialized_str = json.dumps(dataclasses.asdict(obj))
    data = json.loads(serialized_str)
    fields = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in fields}
    return cls(**kwargs)


def _roundtrip_with_tuples(obj: Any, cls: type, tuple_fields: tuple[str, ...] = ()) -> Any:
    serialized_str = json.dumps(dataclasses.asdict(obj))
    data = json.loads(serialized_str)
    fields = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in fields}
    for name in tuple_fields:
        if name in kwargs and isinstance(kwargs[name], list):
            kwargs[name] = tuple(kwargs[name])
    return cls(**kwargs)


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
    assert node_user_a != node_user_b
    restored = _roundtrip(node_user_a, ConditionNode)
    assert restored == node_user_a


def test_dependency_edge_carries_provenance() -> None:
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
    assert "CONDITIONAL" not in {status.value for status in AuthorizationStatus}


def test_defeat_path_type_drops_legacy_rollback_release() -> None:
    values = {path_type.value for path_type in DefeatPathType}
    assert "ROLLBACK_RELEASE" not in values
    assert {"DIRECT_SUPERSEDE", "PREREQUISITE_BLOCK", "UNRESOLVED_UNCERTAIN"} <= values


def test_evidence_edge_type_includes_reaffirms() -> None:
    values = {edge_type.value for edge_type in EvidenceEdgeType}
    assert {"BLOCKS", "RELEASES", "SUPERSEDES", "REAFFIRMS", "UNCERTAIN"} == values

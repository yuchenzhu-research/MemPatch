from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from benchmark.retrace_bench.taxonomy_v2 import (
    DomainV2,
    EventSourceV2,
    EventTypeV2,
    TaskProtocolV2,
    ProbeTypeV2,
    MemoryLifecycleOperationTypeV2,
    MemoryStatusV2,
    ReliabilityFailureModeV2,
    StructuredRevisionActionTypeV2,
    TrustLevelV2,
    SplitV2,
)


@dataclass
class EventV2:
    event_id: str
    source: EventSourceV2
    event_type: EventTypeV2
    actor: str
    timestamp: str
    content: Dict[str, Any]
    visibility_scope: str = "public"
    trust_level: TrustLevelV2 = TrustLevelV2.TRUSTED
    related_memory_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryEntryV2:
    memory_id: str
    category: str  # "belief", "fact", "task_state", "file", "tool_trace"
    content: Any
    source_event_ids: List[str] = field(default_factory=list)
    created_at: str = ""
    visibility_scope: str = "public"
    status: MemoryStatusV2 = MemoryStatusV2.AUTHORIZED
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryLifecycleOperationV2:
    time: str
    operation_type: MemoryLifecycleOperationTypeV2
    memory_id: str
    target_memory_id: Optional[str] = None
    evidence_event_ids: List[str] = field(default_factory=list)
    final_status: MemoryStatusV2 = MemoryStatusV2.AUTHORIZED
    private_notes: Optional[str] = None


@dataclass
class HiddenMemoryLifecycleV2:
    operations: List[MemoryLifecycleOperationV2] = field(default_factory=list)


@dataclass
class StructuredRevisionActionV2:
    action_type: StructuredRevisionActionTypeV2
    target_id: str
    replacement_id: Optional[str] = None
    evidence_event_ids: List[str] = field(default_factory=list)
    scope: Optional[str] = None
    rationale: Optional[str] = None


@dataclass
class GoldEvidenceV2:
    supporting_event_ids: List[str] = field(default_factory=list)
    supporting_memory_ids: List[str] = field(default_factory=list)
    reliability_failure_mode: Optional[ReliabilityFailureModeV2] = None


@dataclass
class GoldBehaviorV2:
    answer: Optional[Any] = None
    gold_memory_statuses: Dict[str, MemoryStatusV2] = field(default_factory=dict)
    gold_actions: List[StructuredRevisionActionV2] = field(default_factory=list)
    gold_evidence: GoldEvidenceV2 = field(default_factory=GoldEvidenceV2)


@dataclass
class TaskV2:
    task_id: str
    protocol: TaskProtocolV2
    probe_type: ProbeTypeV2
    prompt: str
    reference_event_ids: List[str] = field(default_factory=list)
    gold_behavior: GoldBehaviorV2 = field(default_factory=GoldBehaviorV2)
    scoring: Dict[str, Any] = field(default_factory=dict)  # holds metric/protocol params
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioMetadataV2:
    schema_version: str = "2.0"
    source_datasets: List[str] = field(default_factory=list)
    annotators: List[Dict[str, Any]] = field(default_factory=list)
    human_audit: Dict[str, Any] = field(default_factory=dict)
    contamination_policy: Dict[str, Any] = field(default_factory=dict)
    license: str = "CC-BY-SA-4.0"
    citation: str = ""
    additional_properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioV2:
    scenario_id: str
    version: str
    domain: DomainV2
    workflow_context: str
    memory_snapshot: List[MemoryEntryV2]
    event_trace: List[EventV2]
    hidden_memory_lifecycle: HiddenMemoryLifecycleV2
    failure_modes: List[ReliabilityFailureModeV2]
    lifecycle_operations: List[MemoryLifecycleOperationV2]
    tasks: List[TaskV2]
    metadata: ScenarioMetadataV2


@dataclass
class ManifestV2:
    dataset_name: str
    version: str
    scenarios: List[str]
    split: Dict[str, List[str]]
    metadata: ScenarioMetadataV2


@dataclass
class PredictionV2:
    scenario_id: str
    task_id: str
    response: Dict[str, Any]  # answer, memory_statuses, actions, inspect
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResultV2:
    scenario_id: str
    task_id: str
    metrics: Dict[str, Any]  # e.g., {"accuracy": 1.0}
    passed: bool
    feedback: Optional[str] = None
    trace: Optional[str] = None


# Dict Conversion Helpers
def _to_dict_value(val: Any) -> Any:
    if isinstance(val, list):
        return [_to_dict_value(v) for v in val]
    elif isinstance(val, dict):
        return {k: _to_dict_value(v) for k, v in val.items()}
    elif hasattr(val, "__dataclass_fields__"):
        return scenario_to_dict(val)
    elif isinstance(val, Enum):
        return val.value
    return val


def scenario_to_dict(obj: Any) -> Dict[str, Any]:
    """Recursively converts a dataclass (e.g. ScenarioV2, ManifestV2) to a dictionary, resolving Enums."""
    res = {}
    for f in obj.__dataclass_fields__:
        val = getattr(obj, f)
        res[f] = _to_dict_value(val)
    return res


def scenario_from_dict(d: Dict[str, Any]) -> ScenarioV2:
    """Reconstructs a ScenarioV2 instance from a dictionary representation."""
    memory_snapshot = []
    for m in d.get("memory_snapshot", []):
        memory_snapshot.append(MemoryEntryV2(
            memory_id=m["memory_id"],
            category=m["category"],
            content=m["content"],
            source_event_ids=m.get("source_event_ids", []),
            created_at=m.get("created_at", ""),
            visibility_scope=m.get("visibility_scope", "public"),
            status=MemoryStatusV2(m["status"]) if "status" in m else MemoryStatusV2.AUTHORIZED,
            confidence=m.get("confidence"),
            metadata=m.get("metadata", {}),
        ))

    event_trace = []
    for e in d.get("event_trace", []):
        event_trace.append(EventV2(
            event_id=e["event_id"],
            source=EventSourceV2(e["source"]),
            event_type=EventTypeV2(e["event_type"]),
            actor=e["actor"],
            timestamp=e["timestamp"],
            content=e["content"],
            visibility_scope=e.get("visibility_scope", "public"),
            trust_level=TrustLevelV2(e["trust_level"]) if "trust_level" in e else TrustLevelV2.TRUSTED,
            related_memory_ids=e.get("related_memory_ids", []),
            metadata=e.get("metadata", {}),
        ))

    ops = []
    for op in d.get("hidden_memory_lifecycle", {}).get("operations", []):
        ops.append(MemoryLifecycleOperationV2(
            time=op["time"],
            operation_type=MemoryLifecycleOperationTypeV2(op["operation_type"]),
            memory_id=op["memory_id"],
            target_memory_id=op.get("target_memory_id"),
            evidence_event_ids=op.get("evidence_event_ids", []),
            final_status=MemoryStatusV2(op["final_status"]) if "final_status" in op else MemoryStatusV2.AUTHORIZED,
            private_notes=op.get("private_notes"),
        ))
    hidden_memory_lifecycle = HiddenMemoryLifecycleV2(operations=ops)

    lifecycle_operations = []
    for op in d.get("lifecycle_operations", []):
        lifecycle_operations.append(MemoryLifecycleOperationV2(
            time=op["time"],
            operation_type=MemoryLifecycleOperationTypeV2(op["operation_type"]),
            memory_id=op["memory_id"],
            target_memory_id=op.get("target_memory_id"),
            evidence_event_ids=op.get("evidence_event_ids", []),
            final_status=MemoryStatusV2(op["final_status"]) if "final_status" in op else MemoryStatusV2.AUTHORIZED,
            private_notes=op.get("private_notes"),
        ))

    tasks = []
    for t in d.get("tasks", []):
        gb_raw = t.get("gold_behavior", {})
        
        # Parse gold_memory_statuses
        gms_raw = gb_raw.get("gold_memory_statuses", {})
        gold_memory_statuses = {k: MemoryStatusV2(v) for k, v in gms_raw.items()}

        # Parse gold_actions
        gold_actions = []
        for ga in gb_raw.get("gold_actions", []):
            gold_actions.append(StructuredRevisionActionV2(
                action_type=StructuredRevisionActionTypeV2(ga["action_type"]),
                target_id=ga["target_id"],
                replacement_id=ga.get("replacement_id"),
                evidence_event_ids=ga.get("evidence_event_ids", []),
                scope=ga.get("scope"),
                rationale=ga.get("rationale"),
            ))

        # Parse gold_evidence
        ge_raw = gb_raw.get("gold_evidence", {})
        gold_evidence = GoldEvidenceV2(
            supporting_event_ids=ge_raw.get("supporting_event_ids", []),
            supporting_memory_ids=ge_raw.get("supporting_memory_ids", []),
            reliability_failure_mode=ReliabilityFailureModeV2(ge_raw["reliability_failure_mode"])
            if ge_raw.get("reliability_failure_mode") else None
        )

        gold_behavior = GoldBehaviorV2(
            answer=gb_raw.get("answer"),
            gold_memory_statuses=gold_memory_statuses,
            gold_actions=gold_actions,
            gold_evidence=gold_evidence
        )

        tasks.append(TaskV2(
            task_id=t["task_id"],
            protocol=TaskProtocolV2(t["protocol"]),
            probe_type=ProbeTypeV2(t["probe_type"]),
            prompt=t["prompt"],
            reference_event_ids=t.get("reference_event_ids", []),
            gold_behavior=gold_behavior,
            scoring=t.get("scoring", {}),
            metadata=t.get("metadata", {}),
        ))

    meta_raw = d.get("metadata", {})
    metadata = ScenarioMetadataV2(
        schema_version=meta_raw.get("schema_version", "2.0"),
        source_datasets=meta_raw.get("source_datasets", []),
        annotators=meta_raw.get("annotators", []),
        human_audit=meta_raw.get("human_audit", {}),
        contamination_policy=meta_raw.get("contamination_policy", {}),
        license=meta_raw.get("license", "CC-BY-SA-4.0"),
        citation=meta_raw.get("citation", ""),
        additional_properties=meta_raw.get("additional_properties", {}),
    )

    return ScenarioV2(
        scenario_id=d["scenario_id"],
        version=d["version"],
        domain=DomainV2(d["domain"]),
        workflow_context=d.get("workflow_context", ""),
        memory_snapshot=memory_snapshot,
        event_trace=event_trace,
        hidden_memory_lifecycle=hidden_memory_lifecycle,
        failure_modes=[ReliabilityFailureModeV2(fm) for fm in d.get("failure_modes", [])],
        lifecycle_operations=lifecycle_operations,
        tasks=tasks,
        metadata=metadata,
    )


def manifest_from_dict(d: Dict[str, Any]) -> ManifestV2:
    meta_raw = d.get("metadata", {})
    metadata = ScenarioMetadataV2(
        schema_version=meta_raw.get("schema_version", "2.0"),
        source_datasets=meta_raw.get("source_datasets", []),
        annotators=meta_raw.get("annotators", []),
        human_audit=meta_raw.get("human_audit", {}),
        contamination_policy=meta_raw.get("contamination_policy", {}),
        license=meta_raw.get("license", "CC-BY-SA-4.0"),
        citation=meta_raw.get("citation", ""),
        additional_properties=meta_raw.get("additional_properties", {}),
    )
    return ManifestV2(
        dataset_name=d["dataset_name"],
        version=d["version"],
        scenarios=d["scenarios"],
        split=d["split"],
        metadata=metadata,
    )

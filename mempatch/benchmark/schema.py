"""Canonical MemPatch-Bench final / schema v1.0 release contracts.

The release format remains plain JSON so rows can be published as JSONL and
loaded by HuggingFace datasets without a custom builder.  These dataclasses are
thin validation/roundtrip helpers around that JSON contract; they do not own
generation, scoring, or model execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from mempatch.benchmark.contracts import (
    DECISIONS,
    FAILURE_MODES,
    MEMORY_OPERATIONS,
    MEMORY_STATUSES,
    normalize_prediction,
    state_list_to_map,
)
from mempatch.benchmark.leakage import forbidden_paths


CONTRACT_VERSION = "mempatch_bench_schema_v1.0"
PUBLIC_SCHEMA_VERSION = "mempatch_bench_final"
SUPPORTED_PUBLIC_SCHEMA_VERSIONS = {PUBLIC_SCHEMA_VERSION, CONTRACT_VERSION}

FORBIDDEN_PUBLIC_FIELDS = {
    "adjudication_notes",
    "candidate_failure_modes",
    "candidate_memory_operations",
    "difficulty",
    "expected_answer",
    "expected_answer_key_facts",
    "expected_decision",
    "expected_evidence_event_ids",
    "expected_failure_diagnosis",
    "expected_followup_answer",
    "expected_memory_operation",
    "expected_memory_states",
    "failure_mode",
    "generation_metadata",
    "hidden_gold",
    "pattern",
    "resolver_trace",
    "source_pointers",
}


def _errors_required(row: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    return [f"missing {field}" for field in fields if field not in row]


def _ensure_list_of_state_records(value: Any, *, field_name: str) -> list[str]:
    if isinstance(value, dict):
        return [f"{field_name} must be a list of {{memory_id, status}} objects, not a dynamic dict"]
    if not isinstance(value, list):
        return [f"{field_name} must be a list"]
    errors: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"{field_name}[{idx}] must be an object")
            continue
        if "memory_id" not in item:
            errors.append(f"{field_name}[{idx}] missing memory_id")
        if "status" not in item:
            errors.append(f"{field_name}[{idx}] missing status")
        elif item.get("status") not in MEMORY_STATUSES:
            errors.append(f"{field_name}[{idx}] invalid status: {item.get('status')!r}")
    return errors


def validate_public_scenario(row: dict[str, Any]) -> list[str]:
    errors = _errors_required(
        row,
        (
            "scenario_id",
            "schema_version",
            "split",
            "domain",
            "workflow_context",
            "public_input",
            "tasks",
            "output_contract",
        ),
    )
    schema_version = row.get("schema_version")
    if schema_version not in SUPPORTED_PUBLIC_SCHEMA_VERSIONS:
        errors.append(f"unsupported schema_version: {schema_version!r}")
    public_input = row.get("public_input")
    if not isinstance(public_input, dict):
        errors.append("public_input must be an object")
    else:
        if not isinstance(public_input.get("initial_memories"), list):
            errors.append("public_input.initial_memories must be a list")
        if not isinstance(public_input.get("events"), list):
            errors.append("public_input.events must be a list")
        task_text = public_input.get("query") or public_input.get("task_prompt")
        if task_text is not None and not isinstance(task_text, str):
            errors.append("public_input.query/task_prompt must be a string when present")
    if not isinstance(row.get("tasks"), dict):
        errors.append("tasks must be an object")
    forbidden = forbidden_paths(row)
    if forbidden:
        errors.append(f"public row contains forbidden paths: {forbidden}")
    for key in FORBIDDEN_PUBLIC_FIELDS:
        if key in row:
            errors.append(f"public row contains forbidden field: {key}")
    return errors


def validate_private_label(row: dict[str, Any]) -> list[str]:
    errors = _errors_required(
        row,
        (
            "scenario_id",
            "split",
            "expected_decision",
            "expected_memory_operation",
            "expected_memory_states",
            "expected_evidence_event_ids",
            "expected_failure_diagnosis",
        ),
    )
    if row.get("expected_decision") not in DECISIONS:
        errors.append(f"invalid expected_decision: {row.get('expected_decision')!r}")
    if row.get("expected_memory_operation") not in MEMORY_OPERATIONS:
        errors.append(f"invalid expected_memory_operation: {row.get('expected_memory_operation')!r}")
    if row.get("expected_failure_diagnosis") not in FAILURE_MODES:
        errors.append(f"invalid expected_failure_diagnosis: {row.get('expected_failure_diagnosis')!r}")
    errors.extend(_ensure_list_of_state_records(row.get("expected_memory_states"), field_name="expected_memory_states"))
    if not isinstance(row.get("expected_evidence_event_ids"), list):
        errors.append("expected_evidence_event_ids must be a list")
    return errors


def validate_prediction(row: dict[str, Any]) -> list[str]:
    prediction = normalize_prediction(row)
    errors = _errors_required(prediction, ("scenario_id", "parsed"))
    parsed = prediction.get("parsed")
    if not isinstance(parsed, dict):
        return errors + ["parsed must be an object"]
    if "split" in row and row.get("split") is not None and not isinstance(row.get("split"), str):
        errors.append("split must be a string when present")
    for token_field in ("input_tokens", "output_tokens", "retrieved_event_count"):
        value = row.get(token_field)
        if value is not None and not isinstance(value, int):
            errors.append(f"{token_field} must be an integer when present")
    latency = row.get("latency_sec", row.get("latency_seconds"))
    if latency is not None and not isinstance(latency, (int, float)):
        errors.append("latency_sec must be numeric when present")
    return errors


def validate_score_record(row: dict[str, Any]) -> list[str]:
    errors = _errors_required(
        row,
        (
            "scenario_id",
            "split",
            "model",
            "method",
            "schema_valid",
            "parse_failed",
            "exact_state_map",
            "contract_valid_state_success",
            "decision_correct",
            "memory_state_accuracy",
            "evidence_precision",
            "evidence_recall",
            "evidence_f1",
            "diagnosis_correct",
            "strict_joint",
            "unsafe_reuse",
        ),
    )
    for field_name in ("input_tokens", "output_tokens", "total_tokens"):
        value = row.get(field_name)
        if value is not None and not isinstance(value, (int, float)):
            errors.append(f"{field_name} must be numeric when present")
    return errors


def audit_public_private_pair(public_row: dict[str, Any], label_row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if public_row.get("scenario_id") != label_row.get("scenario_id"):
        errors.append(
            f"scenario_id mismatch: public={public_row.get('scenario_id')!r} "
            f"label={label_row.get('scenario_id')!r}"
        )
    public_input = public_row.get("public_input") or {}
    event_ids = {
        str(event.get("event_id"))
        for event in public_input.get("events") or public_input.get("event_trace") or []
        if isinstance(event, dict) and event.get("event_id")
    }
    missing = [
        str(event_id)
        for event_id in label_row.get("expected_evidence_event_ids") or []
        if str(event_id) not in event_ids
    ]
    if missing:
        errors.append(f"expected_evidence_event_ids missing from public events: {missing}")
    errors.extend(validate_public_scenario(public_row))
    errors.extend(validate_private_label(label_row))
    return errors


@dataclass(frozen=True)
class PublicScenario:
    scenario_id: str
    schema_version: str
    split: str
    domain: str
    workflow_context: Any
    public_input: dict[str, Any]
    tasks: dict[str, Any]
    output_contract: dict[str, Any]

    required_fields: ClassVar[tuple[str, ...]] = (
        "scenario_id",
        "schema_version",
        "split",
        "domain",
        "workflow_context",
        "public_input",
        "tasks",
        "output_contract",
    )

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "PublicScenario":
        errors = validate_public_scenario(row)
        if errors:
            raise ValueError("; ".join(errors))
        return cls(**{field: row[field] for field in cls.required_fields})

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "schema_version": self.schema_version,
            "split": self.split,
            "domain": self.domain,
            "workflow_context": self.workflow_context,
            "public_input": self.public_input,
            "tasks": self.tasks,
            "output_contract": self.output_contract,
        }


@dataclass(frozen=True)
class PrivateLabel:
    scenario_id: str
    split: str
    expected_decision: str
    expected_memory_operation: str
    expected_memory_states: list[dict[str, str]]
    expected_evidence_event_ids: list[str]
    expected_failure_diagnosis: str
    expected_answer_key_facts: list[str] = field(default_factory=list)
    difficulty: str | None = None
    failure_mode: str | None = None
    pattern: str | None = None
    memory_operation: str | None = None
    provenance: dict[str, Any] | None = None
    adjudication_notes: str | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "PrivateLabel":
        errors = validate_private_label(row)
        if errors:
            raise ValueError("; ".join(errors))
        return cls(
            scenario_id=str(row["scenario_id"]),
            split=str(row["split"]),
            expected_decision=str(row["expected_decision"]),
            expected_memory_operation=str(row["expected_memory_operation"]),
            expected_memory_states=[
                {"memory_id": str(item["memory_id"]), "status": str(item["status"])}
                for item in row["expected_memory_states"]
            ],
            expected_evidence_event_ids=[str(item) for item in row["expected_evidence_event_ids"]],
            expected_failure_diagnosis=str(row["expected_failure_diagnosis"]),
            expected_answer_key_facts=[str(item) for item in row.get("expected_answer_key_facts") or []],
            difficulty=row.get("difficulty"),
            failure_mode=row.get("failure_mode"),
            pattern=row.get("pattern"),
            memory_operation=row.get("memory_operation") or row.get("expected_memory_operation"),
            provenance=row.get("provenance") or row.get("source"),
            adjudication_notes=row.get("adjudication_notes") or row.get("resolver_trace"),
        )

    def to_dict(self) -> dict[str, Any]:
        row = {
            "scenario_id": self.scenario_id,
            "split": self.split,
            "expected_decision": self.expected_decision,
            "expected_memory_operation": self.expected_memory_operation,
            "expected_memory_states": self.expected_memory_states,
            "expected_evidence_event_ids": self.expected_evidence_event_ids,
            "expected_failure_diagnosis": self.expected_failure_diagnosis,
            "expected_answer_key_facts": self.expected_answer_key_facts,
            "difficulty": self.difficulty,
            "failure_mode": self.failure_mode,
            "pattern": self.pattern,
            "memory_operation": self.memory_operation,
            "provenance": self.provenance,
            "adjudication_notes": self.adjudication_notes,
        }
        return {key: value for key, value in row.items() if value not in (None, [], {})}


@dataclass(frozen=True)
class Prediction:
    scenario_id: str
    model: str | None
    method: str | None
    split: str | None
    raw_response: str | None
    parsed: dict[str, Any]
    parse_error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_sec: float | None = None
    retrieved_event_count: int | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Prediction":
        errors = validate_prediction(row)
        if errors:
            raise ValueError("; ".join(errors))
        normalized = normalize_prediction(row)
        raw_generation = row.get("raw_generation") or {}
        latency = row.get("latency_sec", row.get("latency_seconds"))
        if latency is None:
            latency = raw_generation.get("latency_seconds")
        return cls(
            scenario_id=str(normalized["scenario_id"]),
            model=row.get("model"),
            method=row.get("method"),
            split=row.get("split"),
            raw_response=normalized.get("raw_response"),
            parsed=dict(normalized["parsed"]),
            parse_error=row.get("parse_error"),
            input_tokens=row.get("input_tokens", raw_generation.get("input_tokens")),
            output_tokens=row.get("output_tokens", raw_generation.get("output_tokens")),
            latency_sec=float(latency) if latency is not None else None,
            retrieved_event_count=row.get("retrieved_event_count"),
        )

    def to_dict(self) -> dict[str, Any]:
        row = {
            "scenario_id": self.scenario_id,
            "model": self.model,
            "method": self.method,
            "split": self.split,
            "raw_response": self.raw_response,
            "parsed": self.parsed,
            "parse_error": self.parse_error,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_sec": self.latency_sec,
            "retrieved_event_count": self.retrieved_event_count,
        }
        return {key: value for key, value in row.items() if value is not None}


@dataclass(frozen=True)
class ScoreRecord:
    scenario_id: str
    split: str
    model: str
    method: str
    schema_valid: bool
    parse_failed: bool
    exact_state_map: bool
    contract_valid_state_success: bool
    decision_correct: bool
    memory_state_accuracy: float
    evidence_precision: float
    evidence_recall: float
    evidence_f1: float
    diagnosis_correct: bool
    strict_joint: bool
    unsafe_reuse: bool
    downstream_contamination: bool | None = None
    decision_macro_f1_class: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_sec: float | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ScoreRecord":
        normalized = dict(row)
        normalized.setdefault("parse_failed", bool(row.get("parse_failure")))
        normalized.setdefault(
            "contract_valid_state_success",
            bool(row.get("schema_valid")) and bool(row.get("exact_state_map")),
        )
        normalized.setdefault("decision_macro_f1_class", row.get("decision_f1_class"))
        if "total_tokens" not in normalized:
            input_tokens = normalized.get("input_tokens")
            output_tokens = normalized.get("output_tokens")
            if input_tokens is not None or output_tokens is not None:
                normalized["total_tokens"] = int(input_tokens or 0) + int(output_tokens or 0)
        errors = validate_score_record(normalized)
        if errors:
            raise ValueError("; ".join(errors))
        return cls(
            scenario_id=str(normalized["scenario_id"]),
            split=str(normalized["split"]),
            model=str(normalized["model"]),
            method=str(normalized["method"]),
            schema_valid=bool(normalized["schema_valid"]),
            parse_failed=bool(normalized["parse_failed"]),
            exact_state_map=bool(normalized["exact_state_map"]),
            contract_valid_state_success=bool(normalized["contract_valid_state_success"]),
            decision_correct=bool(normalized["decision_correct"]),
            memory_state_accuracy=float(normalized["memory_state_accuracy"]),
            evidence_precision=float(normalized["evidence_precision"]),
            evidence_recall=float(normalized["evidence_recall"]),
            evidence_f1=float(normalized["evidence_f1"]),
            diagnosis_correct=bool(normalized["diagnosis_correct"]),
            strict_joint=bool(normalized["strict_joint"]),
            unsafe_reuse=bool(normalized["unsafe_reuse"]),
            downstream_contamination=normalized.get("downstream_contamination"),
            decision_macro_f1_class=normalized.get("decision_macro_f1_class"),
            input_tokens=normalized.get("input_tokens"),
            output_tokens=normalized.get("output_tokens"),
            total_tokens=normalized.get("total_tokens"),
            latency_sec=normalized.get("latency_sec"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "split": self.split,
            "model": self.model,
            "method": self.method,
            "schema_valid": self.schema_valid,
            "parse_failed": self.parse_failed,
            "exact_state_map": self.exact_state_map,
            "contract_valid_state_success": self.contract_valid_state_success,
            "decision_correct": self.decision_correct,
            "decision_macro_f1_class": self.decision_macro_f1_class,
            "memory_state_accuracy": self.memory_state_accuracy,
            "evidence_precision": self.evidence_precision,
            "evidence_recall": self.evidence_recall,
            "evidence_f1": self.evidence_f1,
            "diagnosis_correct": self.diagnosis_correct,
            "strict_joint": self.strict_joint,
            "unsafe_reuse": self.unsafe_reuse,
            "downstream_contamination": self.downstream_contamination,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_sec": self.latency_sec,
        }


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    dataset_version: str
    dataset_sha256: str | None
    models: list[str]
    methods: list[str]
    splits: list[str]
    output_root: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "RunManifest":
        return cls(
            run_id=str(row["run_id"]),
            dataset_version=str(row["dataset_version"]),
            dataset_sha256=row.get("dataset_sha256"),
            models=[str(item) for item in row.get("models") or []],
            methods=[str(item) for item in row.get("methods") or []],
            splits=[str(item) for item in row.get("splits") or []],
            output_root=str(row["output_root"]),
            status=str(row["status"]),
            metadata=dict(row.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_version": self.dataset_version,
            "dataset_sha256": self.dataset_sha256,
            "models": self.models,
            "methods": self.methods,
            "splits": self.splits,
            "output_root": self.output_root,
            "status": self.status,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class AggregateRecord:
    split: str
    model: str
    method: str
    n: int
    metrics: dict[str, float]
    status: str = "completed"

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "AggregateRecord":
        metrics = {
            key: float(value)
            for key, value in row.items()
            if key not in {"split", "model", "method", "n", "status"} and isinstance(value, (int, float))
        }
        return cls(
            split=str(row["split"]),
            model=str(row["model"]),
            method=str(row["method"]),
            n=int(row["n"]),
            metrics=metrics,
            status=str(row.get("status") or "completed"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "model": self.model,
            "method": self.method,
            "n": self.n,
            "status": self.status,
            **self.metrics,
        }


def expected_memory_states_as_map(label: dict[str, Any]) -> dict[str, str]:
    """Compatibility helper for consumers that still need a map view."""
    return state_list_to_map(label.get("expected_memory_states"))

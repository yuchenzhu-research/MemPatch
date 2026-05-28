"""Ambiguity-and-Scope Stage A/B feasibility evaluator.

Reuses the AB-1B controlled execution path (``run_case``) and adds the
v3-required scope/abstention/unsupported-revision metrics on top of the
existing fine-grained Stage A diagnostics.

WARNING: This is an internal development diagnostic only.
- Not an official benchmark.
- Not human-validated until per-case ``review_status`` is updated.
- Replay/mock execution is for runner correctness; live execution is
  exploratory development-only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from retracemem.evaluation.controlled_ab import (
    CaseResult,
    InternalDevCase,
    _STATUS_MAP_A_TO_COMPARABLE,
    run_case,
)
from retracemem.methods.contracts import SharedCandidateView
from retracemem.schemas import (
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceNode,
)

SCHEMA_VERSION = "ambiguity_scope_controlled_v0"

REQUIRED_CATEGORIES: tuple[str, ...] = (
    "clear_supersession",
    "clear_prerequisite_blocking",
    "protected_unrelated_belief",
    "temporary_constraint_vs_persistent_preference",
    "current_state_vs_historical_fact",
    "tentative_intention_or_future_possibility",
    "insufficient_evidence_requires_abstention",
    "multi_belief_scope_expansion_trap",
)

CASES_PER_CATEGORY = 4

_NON_USABLE_COMPARABLE = {"NOT_USABLE"}
_USABLE_COMPARABLE = {"USABLE"}
_UNCERTAIN_COMPARABLE = {"UNCERTAIN"}
_STAGE_A_CONFIDENT_COMPARABLE = {"USABLE", "NOT_USABLE"}


@dataclass
class AmbiguityScopeCase:
    """Internal Ambiguity-and-Scope case extending the controlled InternalDevCase."""

    case: InternalDevCase
    category: str
    review_status: str
    source_type: str
    scope_trap: bool
    rationale: str
    protected_belief_ids: tuple[str, ...]
    stale_target_belief_ids: tuple[str, ...]
    abstention_required_belief_ids: tuple[str, ...]
    raw_record: dict[str, Any]


@dataclass
class StageMetricBreakdown:
    """Per-stage scoring counters."""

    total: int = 0
    correct: int = 0
    completed: int = 0
    status_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class CategoryBreakdown:
    """Aggregate per-category counts."""

    stage_a: StageMetricBreakdown = field(default_factory=StageMetricBreakdown)
    stage_b: StageMetricBreakdown = field(default_factory=StageMetricBreakdown)


@dataclass
class AmbiguityScopeMetrics:
    """Aggregate Stage A/B Ambiguity-and-Scope metrics."""

    total_cases: int = 0
    total_belief_decisions: int = 0
    stage_a_overall: StageMetricBreakdown = field(default_factory=StageMetricBreakdown)
    stage_b_overall: StageMetricBreakdown = field(default_factory=StageMetricBreakdown)

    stale_blocking_total: int = 0
    stage_a_stale_blocking_correct: int = 0
    stage_b_stale_blocking_correct: int = 0

    protected_preservation_total: int = 0
    stage_a_protected_preserved: int = 0
    stage_b_protected_preserved: int = 0

    abstention_total: int = 0
    stage_a_abstention_correct: int = 0
    stage_b_abstention_correct: int = 0

    unsupported_revision_total: int = 0
    stage_a_unsupported_revision: int = 0
    stage_b_unsupported_revision: int = 0

    scope_trap_total_cases: int = 0
    scope_trap_protected_total: int = 0
    stage_a_scope_trap_protected: int = 0
    stage_b_scope_trap_protected: int = 0

    by_category: dict[str, CategoryBreakdown] = field(default_factory=dict)

    stage_a_calls: int = 0
    stage_b_calls: int = 0
    stage_a_tokens: int = 0
    stage_b_tokens: int = 0
    stage_a_cache_hits: int = 0
    stage_b_cache_hits: int = 0
    stage_a_latency_ms: float = 0.0
    stage_b_latency_ms: float = 0.0

    execution_errors: int = 0
    parse_errors: int = 0


def _build_view(raw: dict[str, Any]) -> SharedCandidateView:
    evidence_nodes = tuple(
        EvidenceNode(
            evidence_id=e["evidence_id"],
            session_id=e["session_id"],
            timestamp=e.get("timestamp"),
            text=e["text"],
            source_dataset=e["source_dataset"],
            source_pointer=e["source_pointer"],
        )
        for e in raw["evidence_context"]
    )
    new_ev = next(e for e in evidence_nodes if e.evidence_id == raw["new_evidence_id"])

    candidate_beliefs = tuple(
        BeliefNode(
            belief_id=b["belief_id"],
            proposition=b["proposition"],
            source_evidence_ids=tuple(b.get("source_evidence_ids", [])),
        )
        for b in raw["candidate_beliefs"]
    )
    candidate_replacement_beliefs = tuple(
        BeliefNode(
            belief_id=b["belief_id"],
            proposition=b["proposition"],
            source_evidence_ids=tuple(b.get("source_evidence_ids", [])),
        )
        for b in raw.get("candidate_replacement_beliefs", [])
    )
    conditions_by_belief: list[tuple[str, tuple[ConditionNode, ...]]] = []
    for entry in raw.get("candidate_conditions_by_belief", []):
        bid = entry[0]
        conds = tuple(
            ConditionNode(
                condition_id=c["condition_id"],
                scope_id=c["scope_id"],
                text=c["text"],
            )
            for c in entry[1]
        )
        conditions_by_belief.append((bid, conds))
    deps_by_belief: list[tuple[str, tuple[DependencyEdge, ...]]] = []
    for entry in raw.get("dependency_edges_by_belief", []):
        bid = entry[0]
        deps = tuple(
            DependencyEdge(
                edge_id=d["edge_id"],
                belief_id=d["belief_id"],
                condition_id=d["condition_id"],
                inducer=d["inducer"],
                edge_type=d["edge_type"],
            )
            for d in entry[1]
        )
        deps_by_belief.append((bid, deps))

    return SharedCandidateView(
        instance_id=raw["case_id"],
        query_id=f"q_{raw['case_id']}",
        query=raw["query"],
        evidence_context=evidence_nodes,
        new_evidence=new_ev,
        candidate_beliefs=candidate_beliefs,
        candidate_replacement_beliefs=candidate_replacement_beliefs,
        candidate_conditions_by_belief=tuple(conditions_by_belief),
        dependency_edges_by_belief=tuple(deps_by_belief),
    )


def load_dataset(path: str) -> list[AmbiguityScopeCase]:
    """Load and validate the Ambiguity-and-Scope dataset file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if data.get("_schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unexpected schema version: {data.get('_schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )

    cases: list[AmbiguityScopeCase] = []
    seen_ids: set[str] = set()
    for raw in data["cases"]:
        case_id = raw["case_id"]
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen_ids.add(case_id)

        category = raw["category"]
        if category not in REQUIRED_CATEGORIES:
            raise ValueError(
                f"Case {case_id} has category {category!r} which is not in the "
                f"required category set"
            )

        view = _build_view(raw)
        controlled = InternalDevCase(
            case_id=case_id,
            case_type=category,
            description=raw["rationale"],
            view=view,
            expected_stage_a_status=raw["expected_stage_a_status"],
            expected_comparable_status=raw["expected_comparable_status"],
            stage_a_mock_edges=raw["stage_a_mock_edges"],
            stage_b_mock_verdicts=raw["stage_b_mock_verdicts"],
            annotations={
                "protected_beliefs": list(raw.get("protected_belief_ids", [])),
                "stale_target_beliefs": list(raw.get("stale_target_belief_ids", [])),
                "abstention_required_beliefs": list(raw.get("abstention_required_belief_ids", [])),
                "scope_trap": bool(raw.get("scope_trap", False)),
            },
        )

        cases.append(
            AmbiguityScopeCase(
                case=controlled,
                category=category,
                review_status=raw["review_status"],
                source_type=raw["source_type"],
                scope_trap=bool(raw.get("scope_trap", False)),
                rationale=raw["rationale"],
                protected_belief_ids=tuple(raw.get("protected_belief_ids", [])),
                stale_target_belief_ids=tuple(raw.get("stale_target_belief_ids", [])),
                abstention_required_belief_ids=tuple(raw.get("abstention_required_belief_ids", [])),
                raw_record=raw,
            )
        )

    return cases


def validate_dataset_balance(cases: list[AmbiguityScopeCase]) -> None:
    """Enforce category balance and review-status discipline."""
    if len(cases) != len(REQUIRED_CATEGORIES) * CASES_PER_CATEGORY:
        raise ValueError(
            f"Expected {len(REQUIRED_CATEGORIES) * CASES_PER_CATEGORY} cases "
            f"(4 per category), found {len(cases)}"
        )

    per_category: dict[str, int] = {cat: 0 for cat in REQUIRED_CATEGORIES}
    for case in cases:
        per_category[case.category] += 1

    for category, count in per_category.items():
        if count != CASES_PER_CATEGORY:
            raise ValueError(
                f"Category {category} has {count} cases; expected {CASES_PER_CATEGORY}"
            )

    for case in cases:
        if case.review_status not in {
            "model_drafted_pending_human_review",
            "assistant_screened_pending_user_final_review",
            "human_reviewed_locked",
        }:
            raise ValueError(
                f"Case {case.case.case_id} has unsupported review_status "
                f"{case.review_status!r}"
            )


def _actual_stage_a_status(result: Any, belief_id: str) -> str:
    if result is None:
        return ""
    fg = result.provenance.get("fine_grained_statuses", {}) if hasattr(result, "provenance") else {}
    return fg.get(belief_id, "")


def _actual_stage_b_status(result: Any, belief_id: str) -> str:
    if result is None:
        return ""
    for verdict in result.verdicts:
        if verdict.belief_id == belief_id:
            status = verdict.status
            return status.value if hasattr(status, "value") else str(status)
    if belief_id in getattr(result, "authorized_belief_ids", ()):
        return "USABLE"
    if belief_id in getattr(result, "excluded_belief_ids", ()):
        return "NOT_USABLE"
    return ""


def _ensure_category(metrics: AmbiguityScopeMetrics, category: str) -> CategoryBreakdown:
    if category not in metrics.by_category:
        metrics.by_category[category] = CategoryBreakdown()
    return metrics.by_category[category]


def compute_metrics(
    cases: list[AmbiguityScopeCase],
    results: list[CaseResult],
) -> AmbiguityScopeMetrics:
    metrics = AmbiguityScopeMetrics()
    metrics.total_cases = len(cases)

    for case, res in zip(cases, results):
        category = case.category
        cat_breakdown = _ensure_category(metrics, category)
        view = case.case.view

        if res.stage_a_error:
            metrics.execution_errors += 1
            if res.is_stage_a_parse_error:
                metrics.parse_errors += 1
        if res.stage_b_error:
            metrics.execution_errors += 1
            if res.is_stage_b_parse_error:
                metrics.parse_errors += 1

        if case.scope_trap:
            metrics.scope_trap_total_cases += 1

        for belief in view.candidate_beliefs:
            belief_id = belief.belief_id
            expected_a = case.case.expected_stage_a_status.get(belief_id, "")
            expected_comparable = case.case.expected_comparable_status.get(belief_id, "")

            actual_a = _actual_stage_a_status(res.stage_a_result, belief_id)
            actual_b = _actual_stage_b_status(res.stage_b_result, belief_id)
            actual_a_comparable = _STATUS_MAP_A_TO_COMPARABLE.get(actual_a, "")

            metrics.total_belief_decisions += 1

            metrics.stage_a_overall.total += 1
            cat_breakdown.stage_a.total += 1
            if res.stage_a_result is not None:
                metrics.stage_a_overall.completed += 1
                cat_breakdown.stage_a.completed += 1
                metrics.stage_a_overall.status_distribution[actual_a] = (
                    metrics.stage_a_overall.status_distribution.get(actual_a, 0) + 1
                )
                cat_breakdown.stage_a.status_distribution[actual_a] = (
                    cat_breakdown.stage_a.status_distribution.get(actual_a, 0) + 1
                )
                if actual_a_comparable == expected_comparable:
                    metrics.stage_a_overall.correct += 1
                    cat_breakdown.stage_a.correct += 1

            metrics.stage_b_overall.total += 1
            cat_breakdown.stage_b.total += 1
            if res.stage_b_result is not None:
                metrics.stage_b_overall.completed += 1
                cat_breakdown.stage_b.completed += 1
                metrics.stage_b_overall.status_distribution[actual_b] = (
                    metrics.stage_b_overall.status_distribution.get(actual_b, 0) + 1
                )
                cat_breakdown.stage_b.status_distribution[actual_b] = (
                    cat_breakdown.stage_b.status_distribution.get(actual_b, 0) + 1
                )
                if actual_b == expected_comparable:
                    metrics.stage_b_overall.correct += 1
                    cat_breakdown.stage_b.correct += 1

            # Stale-blocking accuracy: target beliefs that must end up NOT_USABLE.
            if belief_id in case.stale_target_belief_ids:
                metrics.stale_blocking_total += 1
                if res.stage_a_result is not None and actual_a_comparable in _NON_USABLE_COMPARABLE:
                    metrics.stage_a_stale_blocking_correct += 1
                if res.stage_b_result is not None and actual_b in _NON_USABLE_COMPARABLE:
                    metrics.stage_b_stale_blocking_correct += 1

            # Protected-belief preservation.
            if belief_id in case.protected_belief_ids:
                metrics.protected_preservation_total += 1
                if case.scope_trap:
                    metrics.scope_trap_protected_total += 1
                if res.stage_a_result is not None and actual_a_comparable in _USABLE_COMPARABLE:
                    metrics.stage_a_protected_preserved += 1
                    if case.scope_trap:
                        metrics.stage_a_scope_trap_protected += 1
                if res.stage_b_result is not None and actual_b in _USABLE_COMPARABLE:
                    metrics.stage_b_protected_preserved += 1
                    if case.scope_trap:
                        metrics.stage_b_scope_trap_protected += 1

            # Abstention accuracy: belief whose gold comparable status is UNCERTAIN.
            if belief_id in case.abstention_required_belief_ids:
                metrics.abstention_total += 1
                if res.stage_a_result is not None and actual_a_comparable in _UNCERTAIN_COMPARABLE:
                    metrics.stage_a_abstention_correct += 1
                if res.stage_b_result is not None and actual_b in _UNCERTAIN_COMPARABLE:
                    metrics.stage_b_abstention_correct += 1

            # Unsupported confident revision: gold UNCERTAIN but method outputs USABLE/NOT_USABLE.
            if expected_comparable == "UNCERTAIN":
                metrics.unsupported_revision_total += 1
                if res.stage_a_result is not None and actual_a_comparable in _STAGE_A_CONFIDENT_COMPARABLE:
                    metrics.stage_a_unsupported_revision += 1
                if res.stage_b_result is not None and actual_b in _STAGE_A_CONFIDENT_COMPARABLE:
                    metrics.stage_b_unsupported_revision += 1

        cost_a = res.stage_a_cost
        tokens_a = cost_a.get("tokens", {})
        calls_a = cost_a.get("calls", {})
        metrics.stage_a_calls += calls_a.get("total", 0)
        metrics.stage_a_tokens += tokens_a.get("total", 0)
        metrics.stage_a_cache_hits += cost_a.get("cache_hits", 0)
        metrics.stage_a_latency_ms += cost_a.get("latency_ms", 0.0)

        cost_b = res.stage_b_cost
        tokens_b = cost_b.get("tokens", {})
        calls_b = cost_b.get("calls", {})
        metrics.stage_b_calls += calls_b.get("total", 0)
        metrics.stage_b_tokens += tokens_b.get("total", 0)
        metrics.stage_b_cache_hits += cost_b.get("cache_hits", 0)
        metrics.stage_b_latency_ms += cost_b.get("latency_ms", 0.0)

    return metrics


def _ratio_str(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}"


def format_report(
    cases: list[AmbiguityScopeCase],
    results: list[CaseResult],
    metrics: AmbiguityScopeMetrics,
) -> dict[str, Any]:
    per_instance: list[dict[str, Any]] = []
    for case, res in zip(cases, results):
        entry: dict[str, Any] = {
            "case_id": case.case.case_id,
            "category": case.category,
            "review_status": case.review_status,
            "scope_trap": case.scope_trap,
            "protected_belief_ids": list(case.protected_belief_ids),
            "stale_target_belief_ids": list(case.stale_target_belief_ids),
            "abstention_required_belief_ids": list(case.abstention_required_belief_ids),
            "stage_a": {},
            "stage_b": {},
        }
        if res.stage_a_result is not None:
            entry["stage_a"] = {
                "fine_grained_statuses": res.stage_a_result.provenance.get("fine_grained_statuses", {}),
                "authorized_belief_ids": list(res.stage_a_result.authorized_belief_ids),
                "excluded_belief_ids": list(res.stage_a_result.excluded_belief_ids),
                "model_call_trace_ids": list(res.stage_a_result.model_call_trace_ids),
                "cost": res.stage_a_result.cost,
                "provenance": res.stage_a_result.provenance,
            }
        else:
            entry["stage_a"] = {
                "error": res.stage_a_error,
                "is_parse_error": res.is_stage_a_parse_error,
                "cost": res.stage_a_cost,
            }

        if res.stage_b_result is not None:
            entry["stage_b"] = {
                "authorized_belief_ids": list(res.stage_b_result.authorized_belief_ids),
                "excluded_belief_ids": list(res.stage_b_result.excluded_belief_ids),
                "model_call_trace_ids": list(res.stage_b_result.model_call_trace_ids),
                "cost": res.stage_b_result.cost,
                "verdicts": [
                    {"belief_id": v.belief_id, "status": v.status.value, "rationale": v.rationale}
                    for v in res.stage_b_result.verdicts
                ],
                "provenance": res.stage_b_result.provenance,
            }
        else:
            entry["stage_b"] = {
                "error": res.stage_b_error,
                "is_parse_error": res.is_stage_b_parse_error,
                "cost": res.stage_b_cost,
            }
        per_instance.append(entry)

    by_category: dict[str, Any] = {}
    for category, breakdown in metrics.by_category.items():
        by_category[category] = {
            "stage_a": {
                "accuracy": _ratio_str(breakdown.stage_a.correct, breakdown.stage_a.total),
                "completed": _ratio_str(breakdown.stage_a.completed, breakdown.stage_a.total),
                "status_distribution": breakdown.stage_a.status_distribution,
            },
            "stage_b": {
                "accuracy": _ratio_str(breakdown.stage_b.correct, breakdown.stage_b.total),
                "completed": _ratio_str(breakdown.stage_b.completed, breakdown.stage_b.total),
                "status_distribution": breakdown.stage_b.status_distribution,
            },
        }

    return {
        "_disclaimer": [
            "INTERNAL DEVELOPMENT FEASIBILITY STUDY ONLY.",
            "NOT an official benchmark.",
            "NOT a final paper result.",
            "NOT human-validated until per-case review_status is updated.",
            "Replay/mock execution is for runner correctness; live execution is exploratory development-only.",
        ],
        "schema_version": SCHEMA_VERSION,
        "aggregate": {
            "total_cases": metrics.total_cases,
            "total_belief_decisions": metrics.total_belief_decisions,
            "overall_comparable_accuracy": {
                "stage_a": _ratio_str(metrics.stage_a_overall.correct, metrics.stage_a_overall.total),
                "stage_b": _ratio_str(metrics.stage_b_overall.correct, metrics.stage_b_overall.total),
            },
            "stale_blocking_accuracy": {
                "stage_a": _ratio_str(metrics.stage_a_stale_blocking_correct, metrics.stale_blocking_total),
                "stage_b": _ratio_str(metrics.stage_b_stale_blocking_correct, metrics.stale_blocking_total),
            },
            "protected_belief_preservation": {
                "stage_a": _ratio_str(metrics.stage_a_protected_preserved, metrics.protected_preservation_total),
                "stage_b": _ratio_str(metrics.stage_b_protected_preserved, metrics.protected_preservation_total),
            },
            "abstention_accuracy": {
                "stage_a": _ratio_str(metrics.stage_a_abstention_correct, metrics.abstention_total),
                "stage_b": _ratio_str(metrics.stage_b_abstention_correct, metrics.abstention_total),
            },
            "unsupported_confident_revision_rate": {
                "stage_a": _ratio_str(metrics.stage_a_unsupported_revision, metrics.unsupported_revision_total),
                "stage_b": _ratio_str(metrics.stage_b_unsupported_revision, metrics.unsupported_revision_total),
            },
            "scope_trap_protected_preservation": {
                "scope_trap_cases": metrics.scope_trap_total_cases,
                "stage_a": _ratio_str(metrics.stage_a_scope_trap_protected, metrics.scope_trap_protected_total),
                "stage_b": _ratio_str(metrics.stage_b_scope_trap_protected, metrics.scope_trap_protected_total),
            },
            "status_distribution": {
                "stage_a": metrics.stage_a_overall.status_distribution,
                "stage_b": metrics.stage_b_overall.status_distribution,
            },
            "observed_cost": {
                "stage_a": {
                    "calls": metrics.stage_a_calls,
                    "tokens": metrics.stage_a_tokens,
                    "cache_hits": metrics.stage_a_cache_hits,
                    "latency_ms": round(metrics.stage_a_latency_ms, 2),
                },
                "stage_b": {
                    "calls": metrics.stage_b_calls,
                    "tokens": metrics.stage_b_tokens,
                    "cache_hits": metrics.stage_b_cache_hits,
                    "latency_ms": round(metrics.stage_b_latency_ms, 2),
                },
            },
            "execution_errors": metrics.execution_errors,
            "parse_errors": metrics.parse_errors,
        },
        "by_category": by_category,
        "per_instance": per_instance,
    }


def run_cases(
    cases: list[AmbiguityScopeCase],
    tmp_dir: str,
    *,
    client_a: Any | None = None,
    client_b: Any | None = None,
    model_id: str = "mock",
    provider: str = "mock",
    stage_a_prompt_version: str = "evidence_edge_prediction_v0",
) -> list[CaseResult]:
    """Execute all loaded cases via the shared controlled track runner."""
    return [
        run_case(
            case.case,
            tmp_dir,
            client_a=client_a,
            client_b=client_b,
            model_id=model_id,
            provider=provider,
            stage_a_prompt_version=stage_a_prompt_version,
        )
        for case in cases
    ]

"""Controlled A/B evaluation layer for internal development cases.

This module implements:
- Case loading/deserialization into SharedCandidateView inputs.
- Replay/mock execution of Stage A and Stage B.
- Metric computation on known case annotations.
- Deterministic JSON-compatible reporting with provenance.

WARNING: This is an internal development protocol check only.
Results are replay/mock execution only, not an official benchmark,
not strict call-budget matched, and make no claim that ReTrace
outperforms DirectJudge.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.methods.contracts import (
    ControlledMethodResult,
    DirectUsabilityStatus,
    SharedCandidateView,
)
from retracemem.methods.controlled_retrace import ControlledReTraceLLM
from retracemem.methods.directjudge import DirectJudgeLLM
from retracemem.providers.base import MockLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.schemas import (
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceNode,
)
from retracemem.verifier.prompt_evidence_edge_verifier import PromptEvidenceEdgeVerifier

_STATUS_MAP_A_TO_COMPARABLE = {
    "AUTHORIZED": "USABLE",
    "BLOCKED": "NOT_USABLE",
    "SUPERSEDED": "NOT_USABLE",
    "UNRESOLVED": "UNCERTAIN",
}


@dataclass
class InternalDevCase:
    """One internal controlled authorization case."""

    case_id: str
    case_type: str
    description: str
    view: SharedCandidateView
    expected_stage_a_status: dict[str, str]
    expected_comparable_status: dict[str, str]
    stage_a_mock_edges: dict[str, list[dict[str, Any]]]
    stage_b_mock_verdicts: list[dict[str, Any]]
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    """Paired Stage A + Stage B result for one case."""

    case_id: str
    case_type: str
    stage_a_result: ControlledMethodResult | None = None
    stage_b_result: ControlledMethodResult | None = None
    stage_a_error: str | None = None
    stage_b_error: str | None = None


@dataclass
class AggregateMetrics:
    """Aggregate metrics over all internal development cases."""

    total_cases: int = 0
    total_belief_decisions: int = 0
    stage_a_correct: int = 0
    stage_a_total: int = 0
    stage_b_correct: int = 0
    stage_b_total: int = 0
    stage_a_status_breakdown: dict[str, int] = field(default_factory=dict)
    stage_b_verdict_breakdown: dict[str, int] = field(default_factory=dict)
    obsolete_misuse_count: int = 0
    obsolete_misuse_total: int = 0
    protected_belief_preserved_count: int = 0
    protected_belief_total: int = 0
    rollback_recovery_count: int = 0
    rollback_recovery_total: int = 0
    stage_a_calls: int = 0
    stage_a_tokens: int = 0
    stage_a_cache_hits: int = 0
    stage_a_latency_ms: float = 0.0
    stage_b_calls: int = 0
    stage_b_tokens: int = 0
    stage_b_cache_hits: int = 0
    stage_b_latency_ms: float = 0.0
    execution_errors: int = 0
    parse_errors: int = 0


def load_cases(path: str) -> list[InternalDevCase]:
    """Load internal dev cases from JSON file into structured objects."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    cases: list[InternalDevCase] = []
    for raw in data["cases"]:
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
        new_ev_id = raw["new_evidence_id"]
        new_evidence = next(e for e in evidence_nodes if e.evidence_id == new_ev_id)

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
        for item in raw.get("candidate_conditions_by_belief", []):
            bid = item[0]
            conds = tuple(
                ConditionNode(
                    condition_id=c["condition_id"],
                    scope_id=c["scope_id"],
                    text=c["text"],
                )
                for c in item[1]
            )
            conditions_by_belief.append((bid, conds))

        deps_by_belief: list[tuple[str, tuple[DependencyEdge, ...]]] = []
        for item in raw.get("dependency_edges_by_belief", []):
            bid = item[0]
            deps = tuple(
                DependencyEdge(
                    edge_id=d["edge_id"],
                    belief_id=d["belief_id"],
                    condition_id=d["condition_id"],
                    inducer=d["inducer"],
                    edge_type=d["edge_type"],
                )
                for d in item[1]
            )
            deps_by_belief.append((bid, deps))

        view = SharedCandidateView(
            instance_id=raw["case_id"],
            query_id=f"q_{raw['case_id']}",
            query=raw["query"],
            evidence_context=evidence_nodes,
            new_evidence=new_evidence,
            candidate_beliefs=candidate_beliefs,
            candidate_replacement_beliefs=candidate_replacement_beliefs,
            candidate_conditions_by_belief=tuple(conditions_by_belief),
            dependency_edges_by_belief=tuple(deps_by_belief),
        )

        cases.append(InternalDevCase(
            case_id=raw["case_id"],
            case_type=raw["case_type"],
            description=raw["description"],
            view=view,
            expected_stage_a_status=raw["expected_stage_a_status"],
            expected_comparable_status=raw["expected_comparable_status"],
            stage_a_mock_edges=raw["stage_a_mock_edges"],
            stage_b_mock_verdicts=raw["stage_b_mock_verdicts"],
            annotations=raw.get("annotations", {}),
        ))

    return cases


def _build_stage_a_mock_response(edges: list[dict[str, Any]]) -> str:
    """Build a mock JSON response matching PromptEvidenceEdgeVerifier parse format."""
    return json.dumps({"edges": edges})


def _build_stage_b_mock_response(verdicts: list[dict[str, Any]]) -> str:
    """Build a mock JSON response matching DirectJudgeLLM parse format."""
    return json.dumps({"verdicts": verdicts})


def run_case(case: InternalDevCase, tmp_dir: str) -> CaseResult:
    """Execute one internal case through both Stage A and Stage B."""
    result = CaseResult(case_id=case.case_id, case_type=case.case_type)

    # --- Stage A ---
    try:
        # Stage A calls the verifier once per candidate belief.
        # We need to provide ordered mock responses for each belief.
        stage_a_responses: list[str] = []
        for belief in case.view.candidate_beliefs:
            edges = case.stage_a_mock_edges.get(belief.belief_id, [])
            stage_a_responses.append(_build_stage_a_mock_response(edges))

        call_idx = [0]
        original_responses = stage_a_responses

        class _OrderedMockA(MockLLMProvider):
            def generate(self, prompt: str, **kwargs: Any):  # type: ignore[override]
                idx = min(call_idx[0], len(original_responses) - 1)
                self.default_response = original_responses[idx]
                call_idx[0] += 1
                return super().generate(prompt, **kwargs)

        mock_a = _OrderedMockA(default_response=original_responses[0] if original_responses else '{"edges": []}')
        cache_a = JSONLCache(os.path.join(tmp_dir, f"{case.case_id}_a.jsonl"))
        client_a = CachedLLMClient(cache=cache_a, provider_client=mock_a)
        verifier = PromptEvidenceEdgeVerifier(client=client_a, model_id="mock", provider="mock")
        runner_a = ControlledReTraceLLM(edge_verifier=verifier, client=client_a)
        result.stage_a_result = runner_a.run(case.view)
    except Exception as exc:
        result.stage_a_error = f"{type(exc).__name__}: {exc}"

    # --- Stage B ---
    try:
        stage_b_response = _build_stage_b_mock_response(case.stage_b_mock_verdicts)
        mock_b = MockLLMProvider(default_response=stage_b_response)
        cache_b = JSONLCache(os.path.join(tmp_dir, f"{case.case_id}_b.jsonl"))
        client_b = CachedLLMClient(cache=cache_b, provider_client=mock_b)
        judge = DirectJudgeLLM(client=client_b, model_id="mock", provider="mock")
        result.stage_b_result = judge.judge(case.view)
    except Exception as exc:
        result.stage_b_error = f"{type(exc).__name__}: {exc}"

    return result


def compute_metrics(
    cases: list[InternalDevCase],
    results: list[CaseResult],
) -> AggregateMetrics:
    """Compute aggregate metrics from case annotations and results."""
    m = AggregateMetrics()
    m.total_cases = len(cases)

    for case, res in zip(cases, results):
        if res.stage_a_error:
            m.execution_errors += 1
        if res.stage_b_error:
            m.execution_errors += 1

        # Stage A metrics
        if res.stage_a_result is not None:
            prov = res.stage_a_result.provenance
            fg = prov.get("fine_grained_statuses", {})
            for bid, expected_status in case.expected_stage_a_status.items():
                m.stage_a_total += 1
                m.total_belief_decisions += 1
                actual = fg.get(bid, "")
                # Count status breakdown
                m.stage_a_status_breakdown[actual] = m.stage_a_status_breakdown.get(actual, 0) + 1
                if actual == expected_status:
                    m.stage_a_correct += 1
                # Obsolete misuse: expected NOT_USABLE but got AUTHORIZED
                expected_comp = case.expected_comparable_status.get(bid, "")
                if expected_comp == "NOT_USABLE":
                    m.obsolete_misuse_total += 1
                    actual_comp = _STATUS_MAP_A_TO_COMPARABLE.get(actual, "")
                    if actual_comp == "USABLE":
                        m.obsolete_misuse_count += 1
            # Cost accounting
            cost = res.stage_a_result.cost
            tokens = cost.get("tokens", {})
            calls = cost.get("calls", {})
            m.stage_a_tokens += tokens.get("total", 0)
            m.stage_a_calls += sum(calls.values()) if calls else 0
            m.stage_a_cache_hits += cost.get("cache_hits", 0)
            m.stage_a_latency_ms += cost.get("latency_ms", 0.0)

        # Stage B metrics
        if res.stage_b_result is not None:
            for bid, expected_status in case.expected_comparable_status.items():
                m.stage_b_total += 1
                # Find actual Stage B verdict
                actual_b = ""
                for v in res.stage_b_result.verdicts:
                    if v.belief_id == bid:
                        actual_b = v.status.value if hasattr(v.status, "value") else str(v.status)
                        break
                if not actual_b:
                    if bid in res.stage_b_result.authorized_belief_ids:
                        actual_b = "USABLE"
                    elif bid in res.stage_b_result.excluded_belief_ids:
                        actual_b = "NOT_USABLE"
                m.stage_b_verdict_breakdown[actual_b] = m.stage_b_verdict_breakdown.get(actual_b, 0) + 1
                if actual_b == expected_status:
                    m.stage_b_correct += 1
            # Cost accounting
            cost = res.stage_b_result.cost
            tokens = cost.get("tokens", {})
            calls = cost.get("calls", {})
            m.stage_b_tokens += tokens.get("total", 0)
            m.stage_b_calls += sum(calls.values()) if calls else 0
            m.stage_b_cache_hits += cost.get("cache_hits", 0)
            m.stage_b_latency_ms += cost.get("latency_ms", 0.0)

        # Protected-belief preservation
        protected = case.annotations.get("protected_beliefs", [])
        for bid in protected:
            m.protected_belief_total += 1
            if res.stage_a_result is not None:
                fg = res.stage_a_result.provenance.get("fine_grained_statuses", {})
                if fg.get(bid) == "AUTHORIZED":
                    m.protected_belief_preserved_count += 1

        # Rollback recovery
        if case.annotations.get("rollback_case", False):
            for bid, expected in case.expected_stage_a_status.items():
                if expected == "AUTHORIZED":
                    m.rollback_recovery_total += 1
                    if res.stage_a_result is not None:
                        fg = res.stage_a_result.provenance.get("fine_grained_statuses", {})
                        if fg.get(bid) == "AUTHORIZED":
                            m.rollback_recovery_count += 1

    return m


def format_report(
    metrics: AggregateMetrics,
    results: list[CaseResult],
) -> dict[str, Any]:
    """Format a JSON-compatible report with disclaimers."""
    per_instance = []
    for r in results:
        entry: dict[str, Any] = {
            "case_id": r.case_id,
            "case_type": r.case_type,
        }
        if r.stage_a_result is not None:
            entry["stage_a"] = {
                "method_name": r.stage_a_result.method_name,
                "authorized_belief_ids": list(r.stage_a_result.authorized_belief_ids),
                "excluded_belief_ids": list(r.stage_a_result.excluded_belief_ids),
                "model_call_trace_ids": list(r.stage_a_result.model_call_trace_ids),
                "cost": r.stage_a_result.cost,
                "provenance": r.stage_a_result.provenance,
            }
        if r.stage_a_error:
            entry["stage_a_error"] = r.stage_a_error
        if r.stage_b_result is not None:
            entry["stage_b"] = {
                "method_name": r.stage_b_result.method_name,
                "authorized_belief_ids": list(r.stage_b_result.authorized_belief_ids),
                "excluded_belief_ids": list(r.stage_b_result.excluded_belief_ids),
                "model_call_trace_ids": list(r.stage_b_result.model_call_trace_ids),
                "cost": r.stage_b_result.cost,
                "provenance": r.stage_b_result.provenance,
                "verdicts": [
                    {"belief_id": v.belief_id, "status": v.status.value, "rationale": v.rationale}
                    for v in r.stage_b_result.verdicts
                ],
            }
        if r.stage_b_error:
            entry["stage_b_error"] = r.stage_b_error
        per_instance.append(entry)

    return {
        "_disclaimer": [
            "INTERNAL DEVELOPMENT PROTOCOL CHECK ONLY.",
            "Replay/mock execution only.",
            "NOT an official benchmark.",
            "NOT strict call-budget matched.",
            "NO claim that ReTrace outperforms DirectJudge.",
        ],
        "aggregate": {
            "total_cases": metrics.total_cases,
            "total_belief_decisions": metrics.total_belief_decisions,
            "stage_a_accuracy": f"{metrics.stage_a_correct}/{metrics.stage_a_total}",
            "stage_b_accuracy": f"{metrics.stage_b_correct}/{metrics.stage_b_total}",
            "stage_a_status_breakdown": metrics.stage_a_status_breakdown,
            "stage_b_verdict_breakdown": metrics.stage_b_verdict_breakdown,
            "obsolete_misuse": f"{metrics.obsolete_misuse_count}/{metrics.obsolete_misuse_total}",
            "protected_belief_preserved": f"{metrics.protected_belief_preserved_count}/{metrics.protected_belief_total}",
            "rollback_recovery": f"{metrics.rollback_recovery_count}/{metrics.rollback_recovery_total}",
            "unsupported_revision_rate": "NOT YET OPERATIONALIZED in AB-1B. Requires explicit annotation of valid defeat-path structure in case gold fields and unambiguous denominator definition.",
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
        "per_instance": per_instance,
    }

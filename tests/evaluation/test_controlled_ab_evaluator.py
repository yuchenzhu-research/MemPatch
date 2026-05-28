"""Tests for AB-1B controlled A/B evaluator.

Covers:
- Case loading/deserialization into valid SharedCandidateView objects.
- Both methods execute on internal cases without live calls.
- Metric computation on known toy outcomes.
- Stage A/B comparable-status mapping.
- Protected-belief preservation handling.
- Rollback/release recovery handling.
- Observed cost is reported separately, not treated as matched.
- Generated output is JSON-compatible.
- Failures or malformed records are surfaced rather than silently dropped.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from retracemem.evaluation.controlled_ab import (
    AggregateMetrics,
    CaseResult,
    InternalDevCase,
    _STATUS_MAP_A_TO_COMPARABLE,
    compute_metrics,
    format_report,
    load_cases,
    run_case,
)
from retracemem.methods.contracts import SharedCandidateView

_CASES_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    "data",
    "internal_dev",
    "controlled_ab_cases.json",
)


class TestCaseLoading:
    """Test loading/deserialization of internal dev cases."""

    def test_load_cases_returns_list(self) -> None:
        cases = load_cases(_CASES_PATH)
        assert isinstance(cases, list)
        assert len(cases) == 6

    def test_each_case_has_valid_view(self) -> None:
        cases = load_cases(_CASES_PATH)
        for case in cases:
            assert isinstance(case.view, SharedCandidateView)
            assert case.view.view_fingerprint != ""
            assert len(case.view.view_fingerprint) == 64

    def test_case_types_covered(self) -> None:
        cases = load_cases(_CASES_PATH)
        types = {c.case_type for c in cases}
        assert "direct_supersession" in types
        assert "prerequisite_blocking" in types
        assert "protected_unrelated_belief" in types
        assert "uncertainty" in types
        assert "release_rollback" in types
        assert "rejected_proposal_audit" in types

    def test_new_evidence_is_in_context(self) -> None:
        cases = load_cases(_CASES_PATH)
        for case in cases:
            ev_ids = [e.evidence_id for e in case.view.evidence_context]
            assert case.view.new_evidence.evidence_id in ev_ids

    def test_dependency_edges_are_valid(self) -> None:
        cases = load_cases(_CASES_PATH)
        for case in cases:
            for bid, deps in case.view.dependency_edges_by_belief:
                for dep in deps:
                    assert dep.edge_type == "REQUIRES"
                    assert dep.belief_id == bid


class TestExecution:
    """Test that both methods execute offline without live calls."""

    def test_all_cases_execute(self) -> None:
        cases = load_cases(_CASES_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            for case in cases:
                result = run_case(case, tmp)
                # At minimum one of the stages should succeed
                assert result.stage_b_result is not None or result.stage_b_error is not None

    def test_stage_b_always_succeeds(self) -> None:
        cases = load_cases(_CASES_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            for case in cases:
                result = run_case(case, tmp)
                assert result.stage_b_result is not None, f"Stage B failed for {case.case_id}: {result.stage_b_error}"

    def test_stage_a_succeeds_on_valid_cases(self) -> None:
        """Stage A should succeed on all cases except the rejected_proposal_audit
        case, which exercises a parse-level rejection for a hallucinated target."""
        cases = load_cases(_CASES_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            for case in cases:
                result = run_case(case, tmp)
                if case.case_type == "rejected_proposal_audit":
                    # Parser rejects the hallucinated condition target;
                    # this IS the auditable behavior being tested.
                    assert result.stage_a_error is not None
                    assert "c_nonexistent" in result.stage_a_error
                else:
                    assert result.stage_a_result is not None, f"Stage A failed for {case.case_id}: {result.stage_a_error}"

    def test_view_fingerprint_preserved_in_stage_a(self) -> None:
        cases = load_cases(_CASES_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            for case in cases:
                result = run_case(case, tmp)
                if result.stage_a_result:
                    assert result.stage_a_result.provenance["view_fingerprint"] == case.view.view_fingerprint

    def test_view_fingerprint_preserved_in_stage_b(self) -> None:
        cases = load_cases(_CASES_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            for case in cases:
                result = run_case(case, tmp)
                if result.stage_b_result:
                    assert result.stage_b_result.provenance["view_fingerprint"] == case.view.view_fingerprint

    def test_stage_a_trace_ids_nonempty(self) -> None:
        cases = load_cases(_CASES_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            for case in cases:
                result = run_case(case, tmp)
                if result.stage_a_result:
                    assert len(result.stage_a_result.model_call_trace_ids) >= 1

    def test_per_instance_cost_reported(self) -> None:
        cases = load_cases(_CASES_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            for case in cases:
                result = run_case(case, tmp)
                if result.stage_a_result:
                    assert "tokens" in result.stage_a_result.cost
                if result.stage_b_result:
                    assert "tokens" in result.stage_b_result.cost


class TestMetrics:
    """Test metric computation logic."""

    def test_metrics_on_loaded_cases(self) -> None:
        cases = load_cases(_CASES_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            results = [run_case(c, tmp) for c in cases]
        metrics = compute_metrics(cases, results)
        assert metrics.total_cases == 6
        assert metrics.total_belief_decisions > 0
        assert metrics.stage_a_total > 0
        assert metrics.stage_b_total > 0

    def test_status_mapping_correct(self) -> None:
        assert _STATUS_MAP_A_TO_COMPARABLE["AUTHORIZED"] == "USABLE"
        assert _STATUS_MAP_A_TO_COMPARABLE["BLOCKED"] == "NOT_USABLE"
        assert _STATUS_MAP_A_TO_COMPARABLE["SUPERSEDED"] == "NOT_USABLE"
        assert _STATUS_MAP_A_TO_COMPARABLE["UNRESOLVED"] == "UNCERTAIN"

    def test_protected_belief_metric(self) -> None:
        cases = load_cases(_CASES_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            results = [run_case(c, tmp) for c in cases]
        metrics = compute_metrics(cases, results)
        # Cases 3 and 6 have protected beliefs
        assert metrics.protected_belief_total >= 2

    def test_rollback_recovery_metric(self) -> None:
        cases = load_cases(_CASES_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            results = [run_case(c, tmp) for c in cases]
        metrics = compute_metrics(cases, results)
        # Case 5 is rollback
        assert metrics.rollback_recovery_total >= 1

    def test_observed_cost_reported_separately(self) -> None:
        cases = load_cases(_CASES_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            results = [run_case(c, tmp) for c in cases]
        metrics = compute_metrics(cases, results)
        # Cost is tracked separately for A and B
        assert metrics.stage_a_tokens >= 0
        assert metrics.stage_b_tokens >= 0
        # They are independent values, not constrained to be equal
        report = format_report(metrics, results)
        assert "stage_a" in report["aggregate"]["observed_cost"]
        assert "stage_b" in report["aggregate"]["observed_cost"]

    def test_execution_errors_surfaced(self) -> None:
        """Errors should be counted, not silently dropped."""
        cases = load_cases(_CASES_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            results = [run_case(c, tmp) for c in cases]
        metrics = compute_metrics(cases, results)
        # The rejected_proposal_audit case has a parser error in Stage A;
        # this should be surfaced in execution_errors, not silently dropped.
        assert metrics.execution_errors == 1


class TestReporting:
    """Test JSON-compatible report generation."""

    def test_report_is_json_serializable(self) -> None:
        cases = load_cases(_CASES_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            results = [run_case(c, tmp) for c in cases]
        metrics = compute_metrics(cases, results)
        report = format_report(metrics, results)
        # Must be JSON-serializable
        serialized = json.dumps(report, ensure_ascii=False)
        parsed = json.loads(serialized)
        assert "_disclaimer" in parsed
        assert "aggregate" in parsed
        assert "per_instance" in parsed

    def test_disclaimer_present(self) -> None:
        cases = load_cases(_CASES_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            results = [run_case(c, tmp) for c in cases]
        metrics = compute_metrics(cases, results)
        report = format_report(metrics, results)
        disclaimers = report["_disclaimer"]
        assert any("INTERNAL" in d for d in disclaimers)
        assert any("NOT an official benchmark" in d for d in disclaimers)
        assert any("NO claim" in d for d in disclaimers)

    def test_per_instance_contains_provenance(self) -> None:
        cases = load_cases(_CASES_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            results = [run_case(c, tmp) for c in cases]
        metrics = compute_metrics(cases, results)
        report = format_report(metrics, results)
        for entry in report["per_instance"]:
            if "stage_a" in entry:
                assert "provenance" in entry["stage_a"]
                assert "view_fingerprint" in entry["stage_a"]["provenance"]
            if "stage_b" in entry:
                assert "provenance" in entry["stage_b"]

    def test_unsupported_revision_rate_deferred(self) -> None:
        cases = load_cases(_CASES_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            results = [run_case(c, tmp) for c in cases]
        metrics = compute_metrics(cases, results)
        report = format_report(metrics, results)
        urr = report["aggregate"]["unsupported_revision_rate"]
        assert "NOT YET OPERATIONALIZED" in urr


class TestEdgeCases:
    """Test edge proposal provenance and rejected edges."""

    def test_rejected_edge_case_surfaces_error(self) -> None:
        """The rejected_proposal_audit case surfaces a parser-level rejection as an error.

        The verifier parser performs strict validation: a BLOCKS edge targeting a
        condition not in the supplied candidate_conditions is rejected before it
        reaches the gate. This is correct AB-1A.5 behavior — the error is surfaced
        in the result rather than silently dropped."""
        cases = load_cases(_CASES_PATH)
        rejected_case = next(c for c in cases if c.case_type == "rejected_proposal_audit")
        with tempfile.TemporaryDirectory() as tmp:
            result = run_case(rejected_case, tmp)
        # Stage A error is surfaced, not silently dropped
        assert result.stage_a_error is not None
        assert "c_nonexistent" in result.stage_a_error
        assert "BLOCKS" in result.stage_a_error
        # Stage B still succeeds (it does not parse edges)
        assert result.stage_b_result is not None
        assert "b_painter" in result.stage_b_result.authorized_belief_ids

    def test_supersession_case_produces_excluded(self) -> None:
        cases = load_cases(_CASES_PATH)
        sup_case = next(c for c in cases if c.case_type == "direct_supersession")
        with tempfile.TemporaryDirectory() as tmp:
            result = run_case(sup_case, tmp)
        assert result.stage_a_result is not None
        assert "b_google_swe" in result.stage_a_result.excluded_belief_ids

    def test_release_case_produces_authorized(self) -> None:
        cases = load_cases(_CASES_PATH)
        rel_case = next(c for c in cases if c.case_type == "release_rollback")
        with tempfile.TemporaryDirectory() as tmp:
            result = run_case(rel_case, tmp)
        assert result.stage_a_result is not None
        assert "b_cycles_to_work" in result.stage_a_result.authorized_belief_ids

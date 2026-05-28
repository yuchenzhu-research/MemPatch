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
        assert "releases_smoke" in types
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
        """Stage A should succeed on all cases since the rejected proposal is now
        designed to be rejected by the gate instead of causing a parse error."""
        cases = load_cases(_CASES_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            for case in cases:
                result = run_case(case, tmp)
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

    def test_rollback_recovery_metric_deferred(self) -> None:
        cases = load_cases(_CASES_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            results = [run_case(c, tmp) for c in cases]
        metrics = compute_metrics(cases, results)
        assert metrics.rollback_recovery_total == 0
        report = format_report(metrics, results)
        assert "NOT YET OPERATIONALIZED" in report["aggregate"]["rollback_recovery"]

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

    def test_execution_errors_surfaced_zero_on_clean_run(self) -> None:
        """On a clean run of the 6 internal cases, there should be zero execution errors."""
        cases = load_cases(_CASES_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            results = [run_case(c, tmp) for c in cases]
        metrics = compute_metrics(cases, results)
        assert metrics.execution_errors == 0


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

    def test_rejected_edge_case_provenance(self) -> None:
        """Verify that the rejected_proposal_audit case succeeds, returns admitted=False and stable gate reason."""
        cases = load_cases(_CASES_PATH)
        rejected_case = next(c for c in cases if c.case_type == "rejected_proposal_audit")
        with tempfile.TemporaryDirectory() as tmp:
            result = run_case(rejected_case, tmp)
        
        # Stage A succeeds
        assert result.stage_a_error is None
        assert result.stage_a_result is not None
        
        # Prove provenance records the rejected proposal
        prov = result.stage_a_result.provenance
        proposals = prov.get("edge_proposals", [])
        assert len(proposals) == 1
        prop = proposals[0]
        assert prop["edge_type"] == "BLOCKS"
        assert prop["target_id"] == "c_valid_license"
        assert prop["admitted"] is False
        assert prop["gate_reason"] == "replacement_belief_id_only_valid_for_supersedes"
        
        # Stage A status remains AUTHORIZED
        assert result.stage_a_result.provenance["fine_grained_statuses"]["b_drive_to_work"] == "AUTHORIZED"
        
        # Stage B still succeeds
        assert result.stage_b_result is not None
        assert "b_drive_to_work" in result.stage_b_result.authorized_belief_ids

    def test_supersession_case_produces_excluded(self) -> None:
        cases = load_cases(_CASES_PATH)
        sup_case = next(c for c in cases if c.case_type == "direct_supersession")
        with tempfile.TemporaryDirectory() as tmp:
            result = run_case(sup_case, tmp)
        assert result.stage_a_result is not None
        assert "b_google_swe" in result.stage_a_result.excluded_belief_ids

    def test_release_case_produces_authorized(self) -> None:
        cases = load_cases(_CASES_PATH)
        rel_case = next(c for c in cases if c.case_id == "dev_release_01")
        with tempfile.TemporaryDirectory() as tmp:
            result = run_case(rel_case, tmp)
        assert result.stage_a_result is not None
        assert "b_cycles_to_work" in result.stage_a_result.authorized_belief_ids


class TestAuditHardening:
    """Test cost accounting correctness, call counts and error capturing."""

    def test_cost_accounting_call_counts_correct(self) -> None:
        """Verify:
        - successful Stage B contributing exactly 1 call.
        - Stage A contributing exactly 1 call per candidate belief on success.
        - aggregate totals match actual semantic invocation counts without double counting.
        """
        cases = load_cases(_CASES_PATH)
        # Choose case 3 (dev_protected_01), which has 2 candidate beliefs
        case = next(c for c in cases if c.case_id == "dev_protected_01")
        assert len(case.view.candidate_beliefs) == 2

        with tempfile.TemporaryDirectory() as tmp:
            res = run_case(case, tmp)

        # Stage B has 1 call
        assert res.stage_b_cost.get("calls", {}).get("total", 0) == 1
        # Stage A has 2 candidate beliefs, so 2 calls
        assert res.stage_a_cost.get("calls", {}).get("total", 0) == 2

        # Check total calls across all cases
        with tempfile.TemporaryDirectory() as tmp:
            results = [run_case(c, tmp) for c in cases]
        metrics = compute_metrics(cases, results)

        # Confirm aggregate call count matches sum of individual case belief counts / judge calls
        expected_a_calls = sum(len(c.view.candidate_beliefs) for c in cases)
        expected_b_calls = len(cases)
        assert metrics.stage_a_calls == expected_a_calls
        assert metrics.stage_b_calls == expected_b_calls
        
        # Verify that calls is not double-counted (since CostAccounting records both success and total)
        assert metrics.stage_a_calls == expected_a_calls

    def test_failed_execution_captures_cost_and_parse_error(self) -> None:
        """Verify that offline parser-error paths contribute observed cost and increment parse_errors."""
        cases = load_cases(_CASES_PATH)
        # Create a deep copy of case 1
        import copy
        case = copy.deepcopy(cases[0])
        # Insert a malformed edge that will fail edge parsing (missing target_id)
        case.stage_a_mock_edges = {
            "b_google_swe": [{"edge_type": "BLOCKS"}]  # missing target_id -> ValueError
        }

        with tempfile.TemporaryDirectory() as tmp:
            res = run_case(case, tmp)

        # Stage A should have failed
        assert res.stage_a_result is None
        assert res.stage_a_error is not None
        assert "ValueError" in res.stage_a_error or "missing target_id" in res.stage_a_error
        assert res.is_stage_a_parse_error is True

        # Cost should still be captured
        assert res.stage_a_cost.get("calls", {}).get("total", 0) == 1

        # Check metrics computation
        metrics = compute_metrics([case], [res])
        assert metrics.execution_errors == 1
        assert metrics.parse_errors == 1
        # Cost is aggregated
        assert metrics.stage_a_calls == 1

    def test_failed_instance_cost_and_error_in_report(self) -> None:
        """Verify that a failed execution serializes cost, error, and parse flags into per-instance report."""
        cases = load_cases(_CASES_PATH)
        import copy
        case = copy.deepcopy(cases[0])
        case.stage_a_mock_edges = {
            "b_google_swe": [{"edge_type": "BLOCKS"}]  # missing target_id -> ValueError
        }

        with tempfile.TemporaryDirectory() as tmp:
            res = run_case(case, tmp)
            metrics = compute_metrics([case], [res])
            report = format_report(metrics, [res])

        # Assert per-instance format
        entry = report["per_instance"][0]
        assert "stage_a" in entry
        assert entry["stage_a"]["error"] is not None
        assert "ValueError" in entry["stage_a"]["error"]
        assert entry["stage_a"]["is_parse_error"] is True
        assert entry["stage_a"]["cost"].get("calls", {}).get("total", 0) == 1

    def test_parse_error_classification(self) -> None:
        """Verify that parse errors are distinguished from execution ValueErrors."""
        cases = load_cases(_CASES_PATH)
        import copy
        
        # Scenario 1: Malformed response ValueError (parser error)
        case_parser_fail = copy.deepcopy(cases[0])
        case_parser_fail.stage_a_mock_edges = {
            "b_google_swe": [{"edge_type": "BLOCKS"}]  # missing target_id -> ValueError
        }
        with tempfile.TemporaryDirectory() as tmp:
            res1 = run_case(case_parser_fail, tmp)
        metrics1 = compute_metrics([case_parser_fail], [res1])
        assert metrics1.execution_errors == 1
        assert metrics1.parse_errors == 1
        
        # Scenario 2: Provider call ValueError (non-parser error)
        from retracemem.schemas import DependencyEdge
        bad_dep = DependencyEdge(
            edge_id="dep-bad",
            belief_id="b_bike_commute",
            condition_id="c_mobility",
            inducer="",  # empty inducer triggers RevisionGate rejection
            edge_type="REQUIRES"
        )
        case_gate_fail = copy.deepcopy(cases[1])
        import dataclasses
        case_gate_fail.view = dataclasses.replace(
            case_gate_fail.view,
            dependency_edges_by_belief=(
                ("b_bike_commute", (bad_dep,)),
            )
        )
        
        with tempfile.TemporaryDirectory() as tmp:
            res2 = run_case(case_gate_fail, tmp)
            
        assert res2.stage_a_result is None
        assert res2.stage_a_error is not None
        assert "rejected by RevisionGate" in res2.stage_a_error
        
        metrics2 = compute_metrics([case_gate_fail], [res2])
        assert metrics2.execution_errors == 1
        assert metrics2.parse_errors == 0  # Should not be classified as a parse error!

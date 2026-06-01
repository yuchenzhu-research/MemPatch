"""Tests for the Ambiguity-and-Scope Stage A/B feasibility evaluator."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pytest

from retracemem.evaluation.ambiguity_scope import (
    HARD_REQUIRED_CATEGORIES,
    HARD_SCHEMA_VERSION,
    REQUIRED_CATEGORIES,
    SCHEMA_VERSION,
    compute_metrics,
    format_report,
    load_dataset,
    run_cases,
    validate_dataset_balance,
)

_DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    "data",
    "retrace_learn",
    "v1",
    "internal_dev",
    "ambiguity_scope_controlled.json",
)
_HARD_DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    "data",
    "retrace_learn",
    "v1",
    "internal_dev",
    "ambiguity_scope_hard_v1.json",
)
_HARD_PILOT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    "configs",
    "ambiguity_scope_hard_v1_pilot.json",
)
_REGRESSION_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    "configs",
    "ambiguity_scope_regression.json",
)


@pytest.fixture(scope="module")
def cases():
    return load_dataset(_DATASET_PATH)


@pytest.fixture(scope="module")
def replay_results(cases):
    with tempfile.TemporaryDirectory() as tmp:
        yield run_cases(cases, tmp)


class TestDatasetSchema:
    def test_schema_version_constant(self):
        with open(_DATASET_PATH, encoding="utf-8") as f:
            data = json.load(f)
        assert data["_schema_version"] == SCHEMA_VERSION

    def test_required_categories_listed(self):
        with open(_DATASET_PATH, encoding="utf-8") as f:
            data = json.load(f)
        assert tuple(data["_required_categories"]) == REQUIRED_CATEGORIES

    def test_cases_load(self, cases):
        assert len(cases) > 0
        for case in cases:
            assert case.case.view.view_fingerprint
            assert case.review_status in {
                "model_drafted_pending_human_review",
                "assistant_screened_pending_user_final_review",
            }
            assert case.source_type == "newly_authored_internal"
            assert case.category in REQUIRED_CATEGORIES

    def test_case_annotation_consistency(self, cases):
        for case in cases:
            expected_keys = set(case.case.expected_stage_a_status.keys())
            comparable_keys = set(case.case.expected_comparable_status.keys())
            assert expected_keys == comparable_keys, (
                f"Case {case.case.case_id} has mismatched expectation belief ids "
                f"between stage A fine-grained and comparable maps"
            )

            view_belief_ids = {b.belief_id for b in case.case.view.candidate_beliefs}
            assert expected_keys == view_belief_ids, (
                f"Case {case.case.case_id} has expectation belief ids that do not "
                f"match candidate beliefs"
            )

            for belief_id in case.protected_belief_ids:
                assert belief_id in view_belief_ids
            for belief_id in case.stale_target_belief_ids:
                assert belief_id in view_belief_ids
            for belief_id in case.abstention_required_belief_ids:
                assert belief_id in view_belief_ids


class TestRunnerDeterminism:
    def test_replay_runs_complete(self, cases, replay_results):
        assert len(replay_results) == len(cases)
        for res in replay_results:
            assert res.stage_b_result is not None
            assert res.stage_a_result is not None or res.stage_a_error is not None

    def test_replay_is_deterministic(self, cases):
        with tempfile.TemporaryDirectory() as tmp1:
            r1 = run_cases(cases, tmp1)
        with tempfile.TemporaryDirectory() as tmp2:
            r2 = run_cases(cases, tmp2)
        for a, b in zip(r1, r2):
            assert a.case_id == b.case_id
            assert (a.stage_a_result is None) == (b.stage_a_result is None)
            assert (a.stage_b_result is None) == (b.stage_b_result is None)
            if a.stage_a_result and b.stage_a_result:
                assert a.stage_a_result.authorized_belief_ids == b.stage_a_result.authorized_belief_ids
                assert a.stage_a_result.excluded_belief_ids == b.stage_a_result.excluded_belief_ids
            if a.stage_b_result and b.stage_b_result:
                assert a.stage_b_result.authorized_belief_ids == b.stage_b_result.authorized_belief_ids
                assert a.stage_b_result.excluded_belief_ids == b.stage_b_result.excluded_belief_ids


class TestMetricsAndReport:
    def test_metrics_compute(self, cases, replay_results):
        metrics = compute_metrics(cases, replay_results)
        assert metrics.total_cases == len(cases)
        assert metrics.total_belief_decisions > 0
        for category in {c.category for c in cases}:
            assert category in metrics.by_category

    def test_report_disclaimer_and_structure(self, cases, replay_results):
        metrics = compute_metrics(cases, replay_results)
        report = format_report(cases, replay_results, metrics)
        disclaimer = report["_disclaimer"]
        assert any("INTERNAL DEVELOPMENT FEASIBILITY STUDY ONLY" in d for d in disclaimer)
        assert any("NOT an official benchmark" in d for d in disclaimer)
        assert any("NOT human-validated" in d for d in disclaimer)
        assert report["schema_version"] == SCHEMA_VERSION
        assert "aggregate" in report
        assert "by_category" in report
        assert "per_instance" in report

    def test_required_metric_keys_present(self, cases, replay_results):
        metrics = compute_metrics(cases, replay_results)
        report = format_report(cases, replay_results, metrics)
        agg = report["aggregate"]
        for key in (
            "overall_comparable_accuracy",
            "stale_blocking_accuracy",
            "protected_belief_preservation",
            "abstention_accuracy",
            "unsupported_confident_revision_rate",
            "scope_trap_protected_preservation",
            "status_distribution",
            "observed_cost",
            "execution_errors",
            "parse_errors",
        ):
            assert key in agg, f"Missing metric: {key}"
        for stage in ("stage_a", "stage_b"):
            assert stage in agg["overall_comparable_accuracy"]
            assert stage in agg["observed_cost"]

    def test_report_is_json_serializable(self, cases, replay_results):
        metrics = compute_metrics(cases, replay_results)
        report = format_report(cases, replay_results, metrics)
        serialized = json.dumps(report, ensure_ascii=False)
        parsed = json.loads(serialized)
        assert parsed["aggregate"]["total_cases"] == len(cases)

    def test_replay_meets_stale_blocking_for_authored_cases(self, cases, replay_results):
        """Replay mock answers reflect the gold; sanity-check stale-blocking accuracy."""
        metrics = compute_metrics(cases, replay_results)
        if metrics.stale_blocking_total > 0:
            assert metrics.stage_a_stale_blocking_correct == metrics.stale_blocking_total
            assert metrics.stage_b_stale_blocking_correct == metrics.stale_blocking_total

    def test_replay_meets_abstention_for_authored_cases(self, cases, replay_results):
        metrics = compute_metrics(cases, replay_results)
        if metrics.abstention_total > 0:
            assert metrics.stage_a_abstention_correct == metrics.abstention_total
            assert metrics.stage_b_abstention_correct == metrics.abstention_total

    def test_replay_no_unsupported_revision(self, cases, replay_results):
        metrics = compute_metrics(cases, replay_results)
        assert metrics.stage_a_unsupported_revision == 0
        assert metrics.stage_b_unsupported_revision == 0

    def test_per_instance_records_review_status(self, cases, replay_results):
        metrics = compute_metrics(cases, replay_results)
        report = format_report(cases, replay_results, metrics)
        for entry in report["per_instance"]:
            assert entry["review_status"] in {
                "model_drafted_pending_human_review",
                "assistant_screened_pending_user_final_review",
            }


class TestBalanceCheck:
    def test_balance_check_passes_for_full_dataset(self, cases):
        validate_dataset_balance(cases)

    def test_balance_check_rejects_partial_dataset(self, cases):
        with pytest.raises(ValueError):
            validate_dataset_balance(cases[:2])

    def test_dataset_has_thirty_two_cases(self, cases):
        assert len(cases) == 32

    def test_each_category_has_four_cases(self, cases):
        counts: dict[str, int] = {}
        for case in cases:
            counts[case.category] = counts.get(case.category, 0) + 1
        for category in REQUIRED_CATEGORIES:
            assert counts.get(category) == 4, (
                f"Category {category} has {counts.get(category, 0)} cases; expected 4"
            )


class TestHardV1Dataset:
    @pytest.fixture(scope="class")
    def hard_cases(self):
        return load_dataset(_HARD_DATASET_PATH)

    def test_hard_schema_and_size(self, hard_cases):
        with open(_HARD_DATASET_PATH, encoding="utf-8") as f:
            data = json.load(f)
        assert data["_schema_version"] == HARD_SCHEMA_VERSION
        assert tuple(data["_required_categories"]) == HARD_REQUIRED_CATEGORIES
        assert len(hard_cases) == 24

    def test_hard_category_balance(self, hard_cases):
        counts: dict[str, int] = {}
        for case in hard_cases:
            counts[case.category] = counts.get(case.category, 0) + 1
        assert counts == {category: 4 for category in HARD_REQUIRED_CATEGORIES}

    def test_hard_required_flags_and_annotations(self, hard_cases):
        for case in hard_cases:
            raw = case.raw_record
            assert raw["challenge_version"] == "hard_v1"
            assert raw["review_status"] == "model_drafted_pending_assistant_and_user_review"
            assert "temporal_reasoning_required" in raw
            assert "premise_resistance" in raw
            belief_ids = {b["belief_id"] for b in raw["candidate_beliefs"]}
            assert set(raw["expected_stage_a_status"]) == belief_ids
            assert set(raw["expected_comparable_status"]) == belief_ids
            assert len(raw["candidate_beliefs"]) >= 3
            if raw["category"] == "dense_semantic_neighborhood_scope_control":
                assert 5 <= len(raw["candidate_beliefs"]) <= 8
            for belief_id in raw["protected_belief_ids"]:
                assert raw["expected_stage_a_status"][belief_id] == "AUTHORIZED"
                assert raw["stage_a_mock_edges"][belief_id] == []

    def test_hard_pilot_one_per_category_and_no_regression_overlap(self, hard_cases):
        with open(_HARD_PILOT_CONFIG_PATH, encoding="utf-8") as f:
            pilot = json.load(f)
        with open(_REGRESSION_CONFIG_PATH, encoding="utf-8") as f:
            regression = json.load(f)
        by_id = {case.case.case_id: case for case in hard_cases}
        selected = pilot["selected_case_ids"]
        assert len(selected) == 6
        assert set(selected).isdisjoint(regression["case_ids"])
        assert {by_id[case_id].category for case_id in selected} == set(HARD_REQUIRED_CATEGORIES)
        assert pilot["prompt_version"] == "evidence_edge_prediction_v1"
        assert pilot["run_type"] == "live_fresh_challenge"

    def test_hard_replay_mock_runner_correctness(self, hard_cases):
        with tempfile.TemporaryDirectory() as tmp:
            results = run_cases(hard_cases, tmp, stage_a_prompt_version="evidence_edge_prediction_v1")
        metrics = compute_metrics(hard_cases, results)
        report = format_report(hard_cases, results, metrics)
        assert report["aggregate"]["total_cases"] == 24
        assert report["aggregate"]["execution_errors"] == 0
        assert report["aggregate"]["parse_errors"] == 0
        total_decisions = sum(len(case.case.view.candidate_beliefs) for case in hard_cases)
        expected = f"{total_decisions}/{total_decisions}"
        assert report["aggregate"]["overall_comparable_accuracy"]["stage_a"] == expected
        assert report["aggregate"]["overall_comparable_accuracy"]["stage_b"] == expected


class TestRunnerScriptShape:
    def test_runner_disclaimer_present_in_script(self):
        script_path = os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            "scripts",
            "run_ambiguity_scope_ab_dev.py",
        )
        text = open(script_path, encoding="utf-8").read()
        assert "Internal development diagnostic only" in text
        assert "Refusing live execution without --live-approved" in text

    def test_runner_supports_pilot_selection_and_review_artifact(self):
        script_path = os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            "scripts",
            "run_ambiguity_scope_ab_dev.py",
        )
        text = open(script_path, encoding="utf-8").read()
        assert "PILOT_CASE_IDS" in text
        assert "--pilot-only" in text
        assert "--case-ids" in text
        assert "--write-review-table" in text
        assert "possible_bias_or_ambiguity" in text
        assert "--run-type" in text
        assert "--stage-a-prompt-version" in text

    def test_pilot_case_ids_cover_categories(self, cases):
        import importlib.util

        script_path = os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            "scripts",
            "run_ambiguity_scope_ab_dev.py",
        )
        spec = importlib.util.spec_from_file_location("run_ambiguity_scope_ab_dev", script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pilot_ids = set(module.PILOT_CASE_IDS)
        pilot_cases = [case for case in cases if case.case.case_id in pilot_ids]
        assert len(pilot_cases) == 8
        assert {case.category for case in pilot_cases} == set(REQUIRED_CATEGORIES)

    def test_runner_records_run_type_and_prompt_version(self, tmp_path):
        script_path = os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            "scripts",
            "run_ambiguity_scope_ab_dev.py",
        )
        result = subprocess.run(
            [
                sys.executable,
                script_path,
                "--mode",
                "replay",
                "--pilot-only",
                "--run-type",
                "replay_correctness",
                "--stage-a-prompt-version",
                "evidence_edge_prediction_v1",
                "--output-dir",
                str(tmp_path),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
        report = json.loads((tmp_path / "ambiguity_scope_report.json").read_text(encoding="utf-8"))
        manifest = json.loads((tmp_path / "ambiguity_scope_manifest.json").read_text(encoding="utf-8"))
        assert report["run_type"] == "replay_correctness"
        assert report["stage_a_prompt_version"] == "evidence_edge_prediction_v1"
        assert manifest["config"]["metadata"]["run_type"] == "replay_correctness"
        assert manifest["config"]["metadata"]["stage_a_prompt_version"] == "evidence_edge_prediction_v1"

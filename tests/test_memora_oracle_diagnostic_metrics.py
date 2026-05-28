from pathlib import Path

from retracemem.adapters.memora_oracle_diagnostic import (
    DIAGNOSTIC_SCORING_LABELS,
    MemoraDiagnosticConfig,
    candidate_roles,
    compute_authorization_diagnostic_metrics,
    compute_stage_authorization_metrics,
    write_report,
)
from retracemem.providers.budget import GlobalBudget
from retracemem.schemas import BeliefNode


def _row(
    *,
    stage_a_authorized=(),
    stage_a_excluded=(),
    stage_b_authorized=(),
    stage_b_excluded=(),
    stage_a_error=None,
    stage_b_error=None,
):
    stage_a = {"error": stage_a_error} if stage_a_error else {
        "authorized_belief_ids": list(stage_a_authorized),
        "excluded_belief_ids": list(stage_a_excluded),
        "provenance": {"fine_grained_statuses": {}},
    }
    stage_b = {"error": stage_b_error} if stage_b_error else {
        "authorized_belief_ids": list(stage_b_authorized),
        "excluded_belief_ids": list(stage_b_excluded),
        "verdict_statuses": {},
        "provenance": {},
    }
    return {
        "candidate_roles": {
            "m1": "memory_presence",
            "f1": "forgetting_absence",
        },
        "stage_a": stage_a,
        "stage_b": stage_b,
    }


def test_candidate_roles_reads_memora_role_metadata():
    beliefs = (
        BeliefNode("m1", "remember", metadata={"memora_role": "memory_presence"}),
        BeliefNode("f1", "forget", metadata={"memora_role": "forgetting_absence"}),
    )
    assert candidate_roles(beliefs) == {
        "m1": "memory_presence",
        "f1": "forgetting_absence",
    }


def test_perfect_memory_presence_and_forgetting_absence_metrics():
    metrics = compute_stage_authorization_metrics([
        _row(stage_a_authorized=("m1",), stage_a_excluded=("f1",))
    ], "stage_a")
    assert metrics["memory_presence_total"] == 1
    assert metrics["memory_presence_authorized"] == 1
    assert metrics["memory_preservation_accuracy"] == 1.0
    assert metrics["forgetting_absence_total"] == 1
    assert metrics["forgetting_absence_excluded"] == 1
    assert metrics["forgetting_suppression_accuracy"] == 1.0
    assert metrics["overall_item_accuracy"] == 1.0
    assert metrics["balanced_authorization_accuracy"] == 1.0


def test_false_retained_forgetting_item_lowers_suppression_accuracy():
    metrics = compute_stage_authorization_metrics([
        _row(stage_a_authorized=("m1", "f1"), stage_a_excluded=())
    ], "stage_a")
    assert metrics["forgetting_absence_total"] == 1
    assert metrics["forgetting_absence_excluded"] == 0
    assert metrics["forgetting_suppression_accuracy"] == 0.0
    assert metrics["overall_item_accuracy"] == 0.5


def test_removed_memory_item_lowers_preservation_accuracy():
    metrics = compute_stage_authorization_metrics([
        _row(stage_a_authorized=(), stage_a_excluded=("m1", "f1"))
    ], "stage_a")
    assert metrics["memory_presence_total"] == 1
    assert metrics["memory_presence_authorized"] == 0
    assert metrics["memory_preservation_accuracy"] == 0.0
    assert metrics["overall_item_accuracy"] == 0.5


def test_stage_a_and_stage_b_metrics_are_independent():
    rows = [
        _row(
            stage_a_authorized=("m1",),
            stage_a_excluded=("f1",),
            stage_b_authorized=("m1", "f1"),
            stage_b_excluded=(),
        )
    ]
    metrics = compute_authorization_diagnostic_metrics(
        rows, [], GlobalBudget(max_calls=10, max_tokens=1000), elapsed_seconds=1.5,
    )
    assert metrics["stage_a"]["overall_item_accuracy"] == 1.0
    assert metrics["stage_b"]["overall_item_accuracy"] == 0.5
    assert metrics["stage_a"]["forgetting_suppression_accuracy"] == 1.0
    assert metrics["stage_b"]["forgetting_suppression_accuracy"] == 0.0
    assert metrics["elapsed_seconds"] == 1.5


def test_missing_or_error_stage_rows_excluded_and_counted_as_execution_errors():
    rows = [
        _row(stage_a_error="boom"),
        {"candidate_roles": {"m1": "memory_presence"}, "stage_a": {}},
        _row(stage_a_authorized=("m1",), stage_a_excluded=("f1",)),
    ]
    metrics = compute_stage_authorization_metrics(rows, "stage_a")
    assert metrics["execution_errors"] == 1
    assert metrics["scored_rows"] == 1
    assert metrics["memory_presence_total"] == 1
    assert metrics["forgetting_absence_total"] == 1


def test_uncertain_statuses_count_from_stage_provenance_and_verdicts():
    row = _row(stage_a_authorized=("m1",), stage_a_excluded=("f1",))
    row["stage_a"]["provenance"] = {"fine_grained_statuses": {"f1": "UNRESOLVED"}}
    row["stage_b"] = {
        "authorized_belief_ids": ["m1"],
        "excluded_belief_ids": ["f1"],
        "verdict_statuses": {"f1": "UNCERTAIN"},
        "provenance": {},
    }
    assert compute_stage_authorization_metrics([row], "stage_a")["uncertain_rate"] == 0.5
    assert compute_stage_authorization_metrics([row], "stage_b")["uncertain_rate"] == 0.5


def test_write_report_retains_non_official_disclaimer_fields(tmp_path: Path):
    rows = [_row(stage_a_authorized=("m1",), stage_a_excluded=("f1",))]
    budget = GlobalBudget(max_calls=10, max_tokens=1000)
    report_path, manifest_path = write_report(
        tmp_path,
        rows,
        [],
        MemoraDiagnosticConfig(reference_root=str(tmp_path)),
        budget,
        verifier_hash="v",
        answer_hash="a",
        personas=["academic_researcher"],
        elapsed_seconds=0.25,
    )
    report = report_path.read_text(encoding="utf-8")
    assert manifest_path.exists()
    assert "MEMORA ORACLE-CONDITIONED AUTHORIZATION DIAGNOSTIC ONLY" in report
    assert '"diagnostic_only": true' in report
    assert '"oracle_conditioned_candidates": true' in report
    assert '"official_end_to_end_result": false' in report
    assert '"official_fama_score": null' in report
    assert '"scoring_type": "oracle_conditioned_authorization_metrics"' in report
    assert '"scoring": "pending"' not in report


def test_metric_computation_does_not_mutate_stage_outputs():
    rows = [_row(stage_a_authorized=("m1",), stage_a_excluded=("f1",))]
    before = repr(rows)
    compute_authorization_diagnostic_metrics(rows, [], GlobalBudget(10, 1000))
    assert repr(rows) == before
    assert DIAGNOSTIC_SCORING_LABELS["official_fama_score"] is None

from __future__ import annotations

import json
from pathlib import Path

from tools.real_seeded.audit_real_seeded import audit
from tools.real_seeded.filter_candidates import filter_rows
from tools.real_seeded.mine_github_candidates import build_candidate
from tools.real_seeded.normalize_real_seeded import normalize_rows


FIXTURE = Path(__file__).parent / "fixtures" / "github_candidate_sample.json"


def sample_candidate() -> dict:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    candidate = build_candidate(
        repo="pydantic/pydantic",
        issue=data["issue"],
        query_group="deprecation",
        query_terms=["deprecated", "migration"],
        comments=data["comments"],
        prs=data["prs"],
        releases=data["releases"],
        docs=data["docs"],
        snapshot_time="2025-01-05T00:00:00Z",
    )
    assert candidate is not None
    return candidate


def test_fixture_based_mining_parser_builds_candidate_without_internet() -> None:
    candidate = sample_candidate()

    assert candidate["source_repo"] == "pydantic/pydantic"
    assert candidate["issue_number"] == 42
    assert len(candidate["raw_events"]) >= 4
    assert "REVISE" in candidate["candidate_memory_operations"]
    assert "stale_memory_reuse" in candidate["candidate_failure_modes"]


def test_filter_accepts_multievent_candidate_and_rejects_one_event_candidate() -> None:
    candidate = sample_candidate()
    accepted, rejected = filter_rows([candidate])

    assert len(accepted) == 1
    assert not rejected
    assert len(accepted[0]["raw_events"]) >= 3

    one_event = {**candidate, "candidate_id": "one_event", "raw_events": candidate["raw_events"][:1]}
    accepted, rejected = filter_rows([one_event])
    assert not accepted
    assert rejected
    assert any("one issue/body event" in reason for reason in rejected[0]["rejection_reasons"])


def test_normalized_public_rows_do_not_include_forbidden_label_fields() -> None:
    accepted, _ = filter_rows([sample_candidate()])
    public_rows, label_rows = normalize_rows(accepted)

    public_text = json.dumps(public_rows[0], sort_keys=True)
    for forbidden in (
        "expected_decision",
        "expected_memory_operation",
        "expected_evidence_event_ids",
        "failure_mode",
        "source_pointers",
        "hidden_gold",
    ):
        assert forbidden not in public_text
    assert label_rows[0]["expected_memory_operation"] == "REVISE"


def test_expected_evidence_ids_exist_in_public_events() -> None:
    accepted, _ = filter_rows([sample_candidate()])
    public_rows, label_rows = normalize_rows(accepted)
    event_ids = {event["event_id"] for event in public_rows[0]["public_input"]["events"]}

    assert set(label_rows[0]["expected_evidence_event_ids"]) <= event_ids


def test_no_token_is_written_to_outputs(tmp_path, monkeypatch) -> None:
    fake_token = "ghp_" + "A" * 36
    monkeypatch.setenv("GITHUB_TOKEN", fake_token)
    accepted, _ = filter_rows([sample_candidate()])
    public_rows, label_rows = normalize_rows(accepted)
    out = tmp_path / "rows.json"
    out.write_text(json.dumps({"public": public_rows, "labels": label_rows}), encoding="utf-8")

    assert fake_token not in out.read_text(encoding="utf-8")


def test_schema_roundtrip_audit_has_no_blocking_violations() -> None:
    accepted, _ = filter_rows([sample_candidate()])
    public_rows, label_rows = normalize_rows(accepted)
    report = audit(public_rows, label_rows)

    assert report["public_rows"] == 1
    assert not report["public_forbidden_field_violations"]
    assert not report["evidence_id_violations"]
    assert not report["sensitive_content_violations"]
    assert not report["state_transition_violations"]


def test_audit_catches_injected_forbidden_field() -> None:
    accepted, _ = filter_rows([sample_candidate()])
    public_rows, label_rows = normalize_rows(accepted)
    public_rows[0]["expected_decision"] = "use_current_memory"
    report = audit(public_rows, label_rows)

    assert report["public_forbidden_field_violations"]


def test_audit_catches_injected_fake_token_and_email() -> None:
    accepted, _ = filter_rows([sample_candidate()])
    public_rows, label_rows = normalize_rows(accepted)
    public_rows[0]["public_input"]["events"][0]["content"] += " ghp_" + "B" * 36
    label_rows[0]["adjudication_notes"] = ["contact " + "maintainer" + "@example.com"]
    report = audit(public_rows, label_rows)

    findings = report["sensitive_content_violations"]
    assert findings
    assert {item["kind"] for row in findings for item in row["findings"]} >= {"token_or_secret", "email"}

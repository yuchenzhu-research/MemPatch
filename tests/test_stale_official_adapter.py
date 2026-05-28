import json
from pathlib import Path

import pytest

from retracemem.adapters.stale_official_adapter import (
    StaleOfficialAdapter,
    parse_record,
)


def _valid_record(uid: str = "uid-1", rtype: str = "T1") -> dict:
    return {
        "uid": uid,
        "M_old": "old fact",
        "M_new": "new fact",
        "explanation": "why",
        "probing_queries": {
            "dim1_query": "q1?",
            "dim2_query": "q2?",
            "dim3_query": "q3?",
        },
        "relevant_session_index": [0],
        "timestamps": ["2025-01-01 09:00", "2025-01-02 09:00"],
        "haystack_session": [["turn 1a", "turn 1b"], ["turn 2"]],
        "type": rtype,
    }


def test_parse_full_file_list_format(tmp_path: Path) -> None:
    path = tmp_path / "T1_T2_400_FULL.json"
    payload = [_valid_record("a", "T1"), _valid_record("b", "T2")]
    path.write_text(json.dumps(payload), encoding="utf-8")
    records = StaleOfficialAdapter(path).load()
    assert len(records) == 2
    assert {r.method_visible.uid for r in records} == {"a", "b"}


def test_method_visible_view_has_only_allowed_fields(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps([_valid_record()]), encoding="utf-8")
    record = StaleOfficialAdapter(path).load()[0]
    visible = record.method_visible
    assert visible.uid == "uid-1"
    assert len(visible.haystack_sessions) == 2
    assert visible.timestamps == ("2025-01-01 09:00", "2025-01-02 09:00")
    assert dict(visible.probing_queries) == {
        "dim1_query": "q1?",
        "dim2_query": "q2?",
        "dim3_query": "q3?",
    }
    assert not hasattr(visible, "m_old")
    assert not hasattr(visible, "m_new")
    assert not hasattr(visible, "explanation")
    assert not hasattr(visible, "relevant_session_index")


def test_evaluator_only_metadata_carries_gold_fields_separately(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps([_valid_record()]), encoding="utf-8")
    record = StaleOfficialAdapter(path).load()[0]
    gold = record.evaluator_only
    assert gold.m_old == "old fact"
    assert gold.m_new == "new fact"
    assert gold.explanation == "why"
    assert gold.relevant_session_index == [0]
    assert gold.type == "T1"


def test_misaligned_sessions_and_timestamps_fail_loudly() -> None:
    record = _valid_record()
    record["timestamps"] = ["only-one"]
    with pytest.raises(ValueError, match="misaligned"):
        parse_record(record, index=0)


def test_missing_probing_query_fails_loudly() -> None:
    record = _valid_record()
    del record["probing_queries"]["dim2_query"]
    with pytest.raises(ValueError, match="dim2_query"):
        parse_record(record, index=0)


def test_invalid_type_fails_loudly() -> None:
    record = _valid_record(rtype="T3")
    with pytest.raises(ValueError, match="unsupported type"):
        parse_record(record, index=0)


def test_stratify_by_type_separates_records(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    payload = [_valid_record("a", "T1"), _valid_record("b", "T2"), _valid_record("c", "T1")]
    path.write_text(json.dumps(payload), encoding="utf-8")
    adapter = StaleOfficialAdapter(path)
    buckets = adapter.stratify_by_type(adapter.load())
    assert {r.method_visible.uid for r in buckets["T1"]} == {"a", "c"}
    assert {r.method_visible.uid for r in buckets["T2"]} == {"b"}


def test_dataset_path_remains_external_and_no_reference_writes(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps([_valid_record()]), encoding="utf-8")
    adapter = StaleOfficialAdapter(path)
    adapter.load()
    assert "reference/" not in str(adapter.dataset_path)


def test_default_dataset_path_points_to_external_directory() -> None:
    adapter = StaleOfficialAdapter()
    assert "data_external/stale_official_frozen" in str(adapter.dataset_path)
    assert adapter.dataset_path.name == "T1_T2_400_FULL.json"

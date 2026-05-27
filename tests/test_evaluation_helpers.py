from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from retracemem.evaluation import (
    CostTracker,
    evaluation_record_from_backend_output,
    read_jsonl,
    records_to_jsonable,
    write_jsonl,
)
from retracemem.schemas import RelationType


@dataclass(frozen=True)
class _NestedRecord:
    relation: RelationType
    path: object


def test_jsonl_round_trip_dataclass_and_enum() -> None:
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "records.jsonl"
        write_jsonl([_NestedRecord(relation=RelationType.SUPPORT, path=path)], path)

        assert read_jsonl(path) == [{"relation": "SUPPORT", "path": str(path)}]


def test_records_to_jsonable_recurses_nested_values() -> None:
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir)

        assert records_to_jsonable({"items": [_NestedRecord(RelationType.BLOCK, path)]}) == {
            "items": [{"relation": "BLOCK", "path": str(path)}]
        }


def test_cost_tracker_totals_merge_and_serialization() -> None:
    tracker = CostTracker()
    tracker.add_tokens("prompt", 10)
    tracker.add_call("search")

    other = CostTracker(tokens={"completion": 4}, calls={"answer": 2})
    tracker.merge(other)

    assert tracker.total_tokens() == 14
    assert tracker.total_calls() == 3
    assert tracker.to_dict() == {
        "tokens": {"prompt": 10, "completion": 4},
        "calls": {"search": 1, "answer": 2},
    }
    assert CostTracker.from_dict(tracker.to_dict()).to_dict() == tracker.to_dict()


def test_cost_tracker_rejects_negative_counts() -> None:
    tracker = CostTracker()

    try:
        tracker.add_tokens("prompt", -1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative token count should fail")

    try:
        tracker.add_call("search", -1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative call count should fail")


def test_evaluation_record_from_backend_output_uses_shared_schema() -> None:
    cost = CostTracker(tokens={"search": 3}, calls={"search": 1})
    retrieved = [{"id": "e1", "text": "Raw evidence", "score": 1.0}]

    record = evaluation_record_from_backend_output(
        query_id="q1",
        method="retrieval_baseline",
        retrieved=retrieved,
        answer="answer shell",
        cost=cost,
        latency_ms=7,
    )

    assert record.query_id == "q1"
    assert record.method == "retrieval_baseline"
    assert record.retrieved_evidence == retrieved
    assert record.authorized_basis == retrieved
    assert record.answer == "answer shell"
    assert record.tokens == {"search": 3}
    assert record.calls == {"search": 1}
    assert record.latency_ms == 7

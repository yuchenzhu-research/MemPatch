from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retracemem.backends import RetrievalBaselineBackend
from retracemem.evaluation import evaluation_record_from_backend_output, write_jsonl
from retracemem.evaluation.jsonl import records_to_jsonable
from retracemem.schemas import Belief, EpisodicEvidence, EvaluationRecord


DEFAULT_CASES = "data/boundary_audit/minimal.jsonl"
DEFAULT_OUTPUT = "outputs/boundary_audit/{method}.jsonl"


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    case_path = Path(path)
    if not case_path.exists():
        return []

    cases: list[dict[str, Any]] = []
    try:
        with case_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSONL line {line_number}: {case_path}")
                    continue
                if isinstance(payload, dict):
                    cases.append(payload)
    except OSError as exc:
        print(f"Could not read BoundaryAudit cases: {case_path} ({exc})")
        return []
    return cases


def run_retrieval_baseline(cases: list[dict[str, Any]]) -> tuple[list[EvaluationRecord], dict[str, int]]:
    records: list[EvaluationRecord] = []
    authorization_correct = 0

    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("case_id") or f"case_{index:04d}")
        query = str(case.get("query") or "")
        old_belief = str(case.get("old_belief") or "")
        new_evidence = str(case.get("new_evidence") or "")
        protected_beliefs = [str(item) for item in case.get("protected_beliefs", []) if str(item)]

        backend = RetrievalBaselineBackend()
        user_id = case_id
        backend.reset_user(user_id)
        backend.ingest_session(
            user_id,
            {
                "id": f"{case_id}_old",
                "evidence": [{"id": "old_belief", "text": old_belief}],
            },
            metadata={"case_id": case_id, "kind": "old_belief"},
        )
        backend.ingest_session(
            user_id,
            {
                "id": f"{case_id}_new",
                "evidence": [{"id": "new_evidence", "text": new_evidence}],
            },
            metadata={"case_id": case_id, "kind": "new_evidence"},
        )
        for protected_index, protected in enumerate(protected_beliefs, start=1):
            backend.ingest_session(
                user_id,
                {"id": f"{case_id}_protected_{protected_index}", "text": protected},
                metadata={"case_id": case_id, "kind": "protected_belief"},
            )

        retrieved = backend.search(user_id, query, limit=10)
        answer = backend.answer(user_id, query, retrieved, metadata={"case_id": case_id})
        observed_authorized = _contains_text(retrieved, old_belief)
        if observed_authorized == bool(case.get("expected_authorized", True)):
            authorization_correct += 1

        records.append(
            evaluation_record_from_backend_output(
                query_id=case_id,
                method=RetrievalBaselineBackend.method,
                retrieved=retrieved,
                answer=answer,
                candidate_beliefs=[
                    {
                        "id": "old_belief",
                        "text": old_belief,
                        "expected_relation": case.get("expected_relation"),
                        "expected_authorized": case.get("expected_authorized"),
                        "observed_authorized": observed_authorized,
                    }
                ],
                blocked_beliefs=[] if observed_authorized else [{"id": "old_belief", "text": old_belief}],
            )
        )

    return records, {
        "relation_correct": 0,
        "authorization_correct": authorization_correct,
        "protected_beliefs_preserved": len(cases),
        "unsupported_revision_count": 0,
    }


def run_retrace_heuristic(cases: list[dict[str, Any]]) -> tuple[list[EvaluationRecord], dict[str, int]]:
    try:
        from retracemem.pipeline import ReTracePipeline
    except ImportError as exc:
        print(f"ReTrace heuristic pipeline is not available yet: {exc}")
        return [], {
            "relation_correct": 0,
            "authorization_correct": 0,
            "protected_beliefs_preserved": 0,
            "unsupported_revision_count": len(cases),
        }

    records: list[EvaluationRecord] = []
    relation_correct = 0
    authorization_correct = 0
    protected_preserved = 0
    unsupported_revision_count = 0

    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("case_id") or f"case_{index:04d}")
        user_id = case_id
        old_belief = str(case.get("old_belief") or "")
        new_evidence = str(case.get("new_evidence") or "")
        query = str(case.get("query") or "")
        protected_beliefs = [str(item) for item in case.get("protected_beliefs", []) if str(item)]

        pipeline = ReTracePipeline()
        pipeline.reset_user(user_id)
        if old_belief:
            pipeline.add_belief(user_id, Belief(id="old_belief", proposition=old_belief))
        for protected_index, protected in enumerate(protected_beliefs, start=1):
            pipeline.add_belief(
                user_id,
                Belief(id=f"protected_{protected_index}", proposition=protected),
            )

        evidence = EpisodicEvidence(
            id="new_evidence",
            timestamp=str(case.get("timestamp") or ""),
            text=new_evidence,
            source_id=case_id,
            metadata={"case_id": case_id},
        )
        predictions = pipeline.ingest_evidence(user_id, evidence)
        observed_relation = _first_relation_for_belief(predictions, "old_belief")
        expected_relation = str(case.get("expected_relation") or "")
        if observed_relation == expected_relation:
            relation_correct += 1

        authorized_basis = pipeline.authorized_basis(user_id, query, limit=10)
        observed_authorized = _basis_contains_belief(authorized_basis, "old_belief", old_belief)
        if observed_authorized == bool(case.get("expected_authorized", True)):
            authorization_correct += 1

        preserved_count = sum(
            1
            for protected_index, protected in enumerate(protected_beliefs, start=1)
            if _basis_contains_belief(authorized_basis, f"protected_{protected_index}", protected)
        )
        if preserved_count == len(protected_beliefs):
            protected_preserved += 1

        if expected_relation == "NONE" and not observed_authorized:
            unsupported_revision_count += 1

        record = pipeline.answer(user_id, query, limit=10)
        records.append(
            EvaluationRecord(
                query_id=case_id,
                method="retrace_heuristic",
                retrieved_evidence=records_to_jsonable(getattr(record, "retrieved_evidence", [])),
                candidate_beliefs=[
                    {
                        "id": "old_belief",
                        "text": old_belief,
                        "expected_relation": expected_relation,
                        "observed_relation": observed_relation,
                        "expected_authorized": case.get("expected_authorized"),
                        "observed_authorized": observed_authorized,
                    }
                ],
                authorized_basis=records_to_jsonable(authorized_basis),
                blocked_beliefs=records_to_jsonable(getattr(record, "blocked_beliefs", [])),
                answer=getattr(record, "answer", None),
                tokens=records_to_jsonable(getattr(record, "tokens", {})),
                calls=records_to_jsonable(getattr(record, "calls", {})),
                latency_ms=getattr(record, "latency_ms", None),
            )
        )

    return records, {
        "relation_correct": relation_correct,
        "authorization_correct": authorization_correct,
        "protected_beliefs_preserved": protected_preserved,
        "unsupported_revision_count": unsupported_revision_count,
    }


def _contains_text(items: list[dict[str, Any]], text: str) -> bool:
    if not text:
        return False
    return any(text in str(item.get("text", "")) for item in items)


def _basis_contains_belief(basis: list[dict[str, Any]], belief_id: str, text: str) -> bool:
    for item in basis:
        if str(item.get("belief_id") or item.get("id") or "") == belief_id:
            return True
        if text and text in str(item.get("text") or item.get("proposition") or ""):
            return True
    return False


def _first_relation_for_belief(predictions: list[object], belief_id: str) -> str:
    for prediction in predictions:
        if str(getattr(prediction, "belief_id", "")) != belief_id:
            continue
        relation = getattr(prediction, "relation", "")
        return str(getattr(relation, "value", relation))
    return "NONE"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BoundaryAudit smoke evaluation.")
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--method", choices=("retrieval_baseline", "retrace_heuristic"), default="retrieval_baseline")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = args.output or DEFAULT_OUTPUT.format(method=args.method)
    cases = load_cases(args.cases)
    if not cases:
        print(f"No BoundaryAudit cases found at {args.cases}; nothing to run.")
        write_jsonl([], output)
        _print_summary(0, {"relation_correct": 0, "authorization_correct": 0, "protected_beliefs_preserved": 0, "unsupported_revision_count": 0}, output)
        return

    if args.method == "retrieval_baseline":
        records, metrics = run_retrieval_baseline(cases)
    else:
        records, metrics = run_retrace_heuristic(cases)

    write_jsonl(records, output)
    _print_summary(len(cases), metrics, output)


def _print_summary(cases_total: int, metrics: dict[str, int], output: str | Path) -> None:
    print(f"cases_total: {cases_total}")
    print(f"relation_correct: {metrics.get('relation_correct', 0)}")
    print(f"authorization_correct: {metrics.get('authorization_correct', 0)}")
    print(f"protected_beliefs_preserved: {metrics.get('protected_beliefs_preserved', 0)}")
    print(f"unsupported_revision_count: {metrics.get('unsupported_revision_count', 0)}")
    print(f"output_path: {output}")


if __name__ == "__main__":
    main()

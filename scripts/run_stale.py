from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retracemem.adapters.stale_adapter import StaleAdapter
from retracemem.backends import RetrievalBaselineBackend
from retracemem.evaluation import evaluation_record_from_backend_output, write_jsonl


DEFAULT_OUTPUT = "outputs/stale/{method}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a ReTrace STALE smoke check.")
    parser.add_argument("--reference-root", default="reference/STALE")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--method", choices=("retrieval_baseline",), default="retrieval_baseline")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    adapter = StaleAdapter(args.reference_root)
    main_files = adapter.discover_main_files()
    output = args.output or DEFAULT_OUTPUT.format(method=args.method)
    if not main_files:
        print(f"No STALE MAIN files found under {args.reference_root}; nothing to run.")
        write_jsonl([], output)
        print("records_written: 0")
        print(f"output_path: {output}")
        return

    samples = adapter.load_records(main_files[0])[: max(args.limit, 0)]
    if not samples:
        print(f"No STALE samples loaded from {main_files[0]}; nothing to run.")
        write_jsonl([], output)
        print("records_written: 0")
        print(f"output_path: {output}")
        return

    backend = RetrievalBaselineBackend()
    records = []
    for sample_index, sample in enumerate(samples, start=1):
        user_id = str(sample.get("sample_id") or sample.get("uid") or f"sample_{sample_index}")
        backend.reset_user(user_id)
        for session in _sessions_for_sample(sample):
            backend.ingest_session(user_id, session, metadata={"sample_id": user_id})

        for query_key, query in _probing_queries(sample).items():
            if not query.strip():
                continue
            retrieved = backend.search(user_id, query, limit=10)
            answer = backend.answer(user_id, query, retrieved, metadata={"sample_id": user_id, "dimension": query_key})
            records.append(
                evaluation_record_from_backend_output(
                    query_id=f"{user_id}:{query_key}",
                    method=args.method,
                    retrieved=retrieved,
                    answer=answer,
                )
            )

    write_jsonl(records, output)
    print(f"source_file: {main_files[0]}")
    print(f"samples_loaded: {len(samples)}")
    print(f"records_written: {len(records)}")
    print(f"output_path: {output}")


def _sessions_for_sample(sample: dict[str, Any]) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    sample_id = str(sample.get("sample_id") or sample.get("uid") or "sample")

    old_memory = str(sample.get("old_memory") or sample.get("M_old") or "")
    if old_memory:
        sessions.append({"id": f"{sample_id}_old_memory", "text": old_memory})

    raw_sessions = sample.get("sessions") or sample.get("haystack_session") or []
    if isinstance(raw_sessions, list):
        for index, raw_session in enumerate(raw_sessions, start=1):
            text = _text_from_benchmark_value(raw_session)
            if text:
                sessions.append({"id": f"{sample_id}_session_{index:04d}", "text": text})

    new_memory = str(sample.get("new_memory") or sample.get("M_new") or "")
    if new_memory:
        sessions.append({"id": f"{sample_id}_new_memory", "text": new_memory})

    return sessions


def _probing_queries(sample: dict[str, Any]) -> dict[str, str]:
    queries = sample.get("probing_queries")
    if not isinstance(queries, dict):
        queries = {}
    return {
        key: str(queries.get(key) or sample.get(key) or "")
        for key in ("dim1_query", "dim2_query", "dim3_query")
    }


def _text_from_benchmark_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "message", "utterance", "value"):
            item = value.get(key)
            if isinstance(item, str):
                return item
        return " ".join(_text_from_benchmark_value(item) for item in value.values()).strip()
    if isinstance(value, list):
        return " ".join(_text_from_benchmark_value(item) for item in value).strip()
    return str(value) if value is not None else ""


if __name__ == "__main__":
    main()

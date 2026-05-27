from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retracemem.adapters.memora_adapter import MemoraAdapter
from retracemem.backends import RetrievalBaselineBackend
from retracemem.evaluation import evaluation_record_from_backend_output, write_jsonl


DEFAULT_OUTPUT = "outputs/memora/{method}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a ReTrace Memora smoke check.")
    parser.add_argument("--reference-root", default="reference/Memora")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--method", choices=("retrieval_baseline",), default="retrieval_baseline")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    adapter = MemoraAdapter(args.reference_root)
    roots = adapter.discover_data_roots()
    output = args.output or DEFAULT_OUTPUT.format(method=args.method)
    if not roots:
        print(f"No Memora data roots found under {args.reference_root}; nothing to run.")
        write_jsonl([], output)
        print("records_written: 0")
        print(f"output_path: {output}")
        return

    root = roots[0]
    period = str(root["period"])
    persona_id = str(root["persona_id"])
    sessions = adapter.load_sessions(period, persona_id)
    questions = adapter.load_evaluation_questions(period, persona_id)[: max(args.limit, 0)]
    if not sessions or not questions:
        print(f"No Memora sessions/questions loaded for {period}/{persona_id}; nothing to run.")
        write_jsonl([], output)
        print("records_written: 0")
        print(f"output_path: {output}")
        return

    backend = RetrievalBaselineBackend()
    user_id = f"{period}:{persona_id}"
    backend.reset_user(user_id)
    for session in sessions:
        backend.ingest_session(
            user_id,
            _session_for_backend(session),
            metadata={"period": period, "persona_id": persona_id},
        )

    records = []
    for question_index, question in enumerate(questions, start=1):
        query = str(question.get("question") or "")
        if not query.strip():
            continue
        question_id = str(question.get("question_id") or f"question_{question_index:04d}")
        retrieved = backend.search(user_id, query, limit=10)
        answer = backend.answer(user_id, query, retrieved, metadata={"question_id": question_id})
        records.append(
            evaluation_record_from_backend_output(
                query_id=question_id,
                method=args.method,
                retrieved=retrieved,
                answer=answer,
            )
        )

    write_jsonl(records, output)
    print(f"data_root: {root['root']}")
    print(f"sessions_loaded: {len(sessions)}")
    print(f"questions_loaded: {len(questions)}")
    print(f"records_written: {len(records)}")
    print(f"output_path: {output}")


def _session_for_backend(session: dict[str, Any]) -> dict[str, Any]:
    messages = []
    conversation = session.get("conversation")
    if isinstance(conversation, list):
        for turn in conversation:
            if not isinstance(turn, dict):
                continue
            content = turn.get("message") or turn.get("content") or turn.get("text")
            if not isinstance(content, str) or not content.strip():
                continue
            role = str(turn.get("speaker") or turn.get("role") or "")
            messages.append({"role": role, "content": content})

    return {
        "id": str(session.get("session_id") or ""),
        "source_id": str(session.get("date") or session.get("session_id") or ""),
        "messages": messages,
        "metadata": session.get("metadata") if isinstance(session.get("metadata"), dict) else {},
    }


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retracemem.adapters.memora_adapter import MemoraAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a Memora sample persona and session data.")
    parser.add_argument("--reference-root", default="reference/Memora")
    parser.add_argument("--period", default=None, help="E.g. weekly or daily")
    parser.add_argument("--persona", default=None, help="E.g. software_engineer")
    parser.add_argument("--limit-sessions", type=int, default=2)
    parser.add_argument("--limit-questions", type=int, default=2)
    args = parser.parse_args()

    adapter = MemoraAdapter(args.reference_root)
    if not adapter.exists():
        print(f"Memora reference directory not found at: {args.reference_root}")
        sys.exit(0)

    roots = adapter.discover_data_roots()
    if not roots:
        print("No Memora persona data roots found.")
        sys.exit(0)

    # Resolve period and persona
    selected_root = None
    if args.period and args.persona:
        for r in roots:
            if r["period"] == args.period and r["persona_id"] == args.persona:
                selected_root = r
                break
        if not selected_root:
            print(f"Could not find data for period '{args.period}' and persona '{args.persona}'.")
            sys.exit(1)
    else:
        selected_root = roots[0]
        print(f"No specific period/persona provided. Defaulting to first found: period={selected_root['period']}, persona={selected_root['persona_id']}")

    period = selected_root["period"]
    persona_id = selected_root["persona_id"]

    # Load raw normalized dicts
    sessions = adapter.load_sessions(period, persona_id)
    questions = adapter.load_evaluation_questions(period, persona_id)

    # Load typed records
    ev_records, q_records = adapter.load_as_records(period, persona_id)

    print("=" * 60)
    print(f"MEMORA INSPECTOR: {persona_id} ({period})")
    print("=" * 60)
    print(f"Total sessions: {len(sessions)}")
    print(f"Total questions: {len(questions)}")
    print(f"Total typed EvidenceRecords: {len(ev_records)}")
    print(f"Total typed QueryRecords: {len(q_records)}")
    print("-" * 60)

    print(f"\n--- Showing first {args.limit_sessions} EvidenceRecords ---")
    for idx, ev in enumerate(ev_records[:args.limit_sessions]):
        print(f"Evidence ID: {ev.evidence_id}")
        print(f"Session ID:  {ev.session_id}")
        print(f"Timestamp:   {ev.timestamp}")
        print(f"Text content:\n{ev.text}")
        print(f"Source Pointer: {ev.source_pointer}")
        print("-" * 40)

    print(f"\n--- Showing first {args.limit_questions} QueryRecords ---")
    for idx, q in enumerate(q_records[:args.limit_questions]):
        print(f"Query ID:   {q.query_id}")
        print(f"Timestamp:  {q.timestamp}")
        print(f"Query Text: {q.query_text}")
        metadata = q.metadata
        print(f"Task Bucket: {metadata.get('task_bucket')}")
        print(f"Evaluation questions: {metadata.get('evaluation', {}).get('total_evaluation_questions', 0)}")
        print("-" * 40)


if __name__ == "__main__":
    main()

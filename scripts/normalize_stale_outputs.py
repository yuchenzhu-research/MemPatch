from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retracemem.schemas import EvaluationRecord
from retracemem.evaluation.jsonl import write_jsonl, records_to_jsonable


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize STALE outputs to ReTrace schema formats.")
    parser.add_argument("--input", help="Path to raw STALE answers JSON file")
    parser.add_argument("--dataset", help="Path to STALE dataset MAIN.json file")
    parser.add_argument("--output", help="Path to output normalized JSONL file")
    parser.add_argument("--method", default="target_model", help="Method name to write to records")
    args = parser.parse_args()

    if not args.input or not Path(args.input).exists():
        print("Input file not specified or does not exist. Generating a mock schema record for illustration.")
        mock_record = EvaluationRecord(
            query_id="mock_sample_1_dim1_query",
            method=args.method,
            answer="This is a mock normalized answer to demonstrate ReTrace schema conversion.",
            retrieved_evidence=[],
            candidate_beliefs=[],
            authorized_basis=[],
            blocked_beliefs=[]
        )
        print("Mock Record:")
        print(json.dumps(records_to_jsonable(mock_record), indent=2))
        return

    # Load inputs
    with open(args.input, "r", encoding="utf-8") as f:
        raw_answers = json.load(f)

    # Let's support both list of dicts and dict of dicts
    if isinstance(raw_answers, dict):
        raw_list = [{"uid": k, **v} for k, v in raw_answers.items()]
    elif isinstance(raw_answers, list):
        raw_list = raw_answers
    else:
        raw_list = []

    normalized_records = []
    for item in raw_list:
        uid = str(item.get("uid") or item.get("sample_id") or "")
        if not uid:
            continue
            
        responses = item.get("target_model_responses") or item.get("responses") or {}

        for dim in ("dim1_query", "dim2_query", "dim3_query"):
            ans = responses.get(dim) or responses.get(dim.replace("query", "response")) or ""
            if not ans and dim not in responses:
                dim_short = dim.split("_")[0]
                ans = responses.get(dim_short) or ""
            
            # Construct evaluation record
            record = EvaluationRecord(
                query_id=f"{uid}_{dim}",
                method=args.method,
                answer=ans,
                retrieved_evidence=[],
                candidate_beliefs=[],
                authorized_basis=[],
                blocked_beliefs=[]
            )
            normalized_records.append(record)

    if args.output:
        write_jsonl(normalized_records, args.output)
        print(f"Successfully normalized {len(normalized_records)} records to {args.output}")
    else:
        print(f"No output path specified. Normalized {len(normalized_records)} records.")
        if normalized_records:
            print("Sample Normalized Record:")
            print(json.dumps(records_to_jsonable(normalized_records[0]), indent=2))


if __name__ == "__main__":
    main()

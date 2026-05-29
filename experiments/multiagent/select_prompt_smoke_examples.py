from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

def select_smoke_examples(review_file: str, confirm_live_run: bool) -> List[Dict[str, Any]]:
    if not os.path.exists(review_file):
        print(f"Error: Review file not found: {review_file}")
        sys.exit(1)
        
    approved_examples: List[Dict[str, Any]] = []
    
    with open(review_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("review_status") == "approved":
                    approved_examples.append(record)
            except Exception as e:
                print(f"Error parsing JSON line: {e}")
                sys.exit(1)
                
    if not approved_examples:
        print("No approved examples available for prompt smoke.")
        sys.exit(1)
        
    if not confirm_live_run:
        print("Error: Live run not confirmed. Please pass --confirm-live-run to execute.")
        sys.exit(1)
        
    # Group by failure_type and select deterministically (e.g. sorted by episode_id, pick first)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for ex in approved_examples:
        f_type = ex.get("failure_type", "unknown")
        grouped.setdefault(f_type, []).append(ex)
        
    selected: List[Dict[str, Any]] = []
    # Deterministically select one per failure type
    for f_type in sorted(grouped.keys()):
        examples_in_group = sorted(grouped[f_type], key=lambda x: x.get("episode_id", ""))
        selected.append(examples_in_group[0])
        
    return selected

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select approved examples for prompt smoke (Packet 4E)"
    )
    parser.add_argument(
        "--review-file",
        type=str,
        default="outputs/stagec_dev_review_queue_70.jsonl",
        help="Path to the human-review JSONL file"
    )
    parser.add_argument(
        "--confirm-live-run",
        action="store_true",
        help="Explicitly confirm this live execution"
    )
    args = parser.parse_args()
    
    selected = select_smoke_examples(args.review_file, args.confirm_live_run)
    print(f"Successfully selected {len(selected)} approved examples for prompt smoke:")
    for ex in selected:
        print(f"  - Episode ID: {ex.get('episode_id')}, Failure Type: {ex.get('failure_type')}")
        
if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retracemem.adapters.stale_adapter import StaleAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="STALE dataset smoke check.")
    parser.add_argument("--reference-root", default="reference/STALE")
    parser.add_argument("--artifact-dir", default=None)
    args = parser.parse_args()

    adapter = StaleAdapter(args.reference_root)
    main_files = adapter.discover_main_files()

    if not main_files:
        print("No STALE MAIN files found under the reference directory. Exiting successfully.")
        sys.exit(0)

    print(f"Discovered STALE MAIN files: {main_files}")
    
    # Load records using the new load_as_records interface
    evidence_records, query_records = adapter.load_as_records(main_files[0])
    
    print(f"Successfully loaded STALE main records:")
    print(f"  Evidence records: {len(evidence_records)}")
    print(f"  Query records: {len(query_records)}")

    # Prepare artifact directory
    artifact_dir_str = args.artifact_dir or os.environ.get(
        "ARTIFACT_DIR", "/Users/yuchenzhu/.gemini/antigravity/brain/6177bcef-bc52-4e7d-bb17-47a0e4ab2f8b"
    )
    artifact_dir = Path(artifact_dir_str)
    
    # If the default dir doesn't exist and we didn't specify it, fallback to outputs/
    if not artifact_dir.exists():
        fallback_dir = Path(__file__).resolve().parents[1] / "outputs" / "artifacts"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir = fallback_dir
        
    print(f"Writing normalized outputs to: {artifact_dir}")
    
    # Convert dataclasses to dict for JSON serialization
    from retracemem.evaluation.jsonl import records_to_jsonable
    ev_dicts = [records_to_jsonable(r) for r in evidence_records]
    q_dicts = [records_to_jsonable(r) for r in query_records]
    
    output_payload = {
        "source_file": str(main_files[0]),
        "evidence_records": ev_dicts,
        "query_records": q_dicts,
    }
    
    output_path = artifact_dir / "stale_smoke_normalized_records.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
        
    print(f"Normalized records successfully written to {output_path}")


if __name__ == "__main__":
    main()

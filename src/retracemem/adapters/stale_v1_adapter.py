from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class StaleV1Adapter:
    """Adapter to format ReTrace evaluation records into STALE's official target_model_responses JSON format."""

    @staticmethod
    def export_to_official_json(
        records: list[Any],
        output_filepath: str | Path,
    ) -> None:
        """Groups EvaluationRecords by sample_id and writes them in the format expected by STALE's evaluator.

        Each record is assumed to have query_id in format "{uid}_{dim_key}" where dim_key
        is e.g. "dim1_query", "dim2_query", or "dim3_query".
        """
        grouped: dict[str, dict[str, str]] = {}

        for record in records:
            # ReTrace records are dataclasses or dicts
            query_id = getattr(record, "query_id", None) or record.get("query_id", "")
            answer = getattr(record, "answer", None) or record.get("answer", "")

            if not query_id or "_" not in query_id:
                continue

            # Check if query_id ends with known probing query suffixes
            if "_dim1_query" in query_id:
                uid = query_id.replace("_dim1_query", "")
                dim_resp_key = "dim1_response"
            elif "_dim2_query" in query_id:
                uid = query_id.replace("_dim2_query", "")
                dim_resp_key = "dim2_response"
            elif "_dim3_query" in query_id:
                uid = query_id.replace("_dim3_query", "")
                dim_resp_key = "dim3_response"
            else:
                parts = query_id.rsplit("_", 1)
                if len(parts) != 2:
                    continue
                uid, dim_key = parts
                dim_resp_key = f"{dim_key}_response" if not dim_key.endswith("_response") else dim_key

            if uid not in grouped:
                grouped[uid] = {
                    "dim1_response": "",
                    "dim2_response": "",
                    "dim3_response": "",
                }

            grouped[uid][dim_resp_key] = answer

        # Build final official list
        official_data = []
        for uid, responses in grouped.items():
            official_data.append({
                "uid": uid,
                "target_model_responses": responses,
            })

        # Write to JSON file
        output_path = Path(output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(official_data, f, indent=2, ensure_ascii=False)

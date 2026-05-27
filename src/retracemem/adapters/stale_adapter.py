from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union


class StaleAdapter:
    """Best-effort loader for STALE MAIN dataset files."""

    def __init__(self, reference_root: str | Path = "reference/STALE") -> None:
        self.reference_root = Path(reference_root)

    def exists(self) -> bool:
        return self.reference_root.exists()

    def discover_main_files(self) -> list[Path]:
        """Return available `*_MAIN.json` files without requiring them to exist."""
        if not self.exists():
            return []
        try:
            return sorted(path for path in self.reference_root.rglob("*_MAIN.json") if path.is_file())
        except OSError:
            return []

    def discover(self) -> list[Path]:
        """Compatibility discovery entrypoint."""
        return self.discover_main_files()

    def load_records(self, path: Optional[Union[str, Path]] = None) -> list[dict[str, Any]]:
        """Load and normalize records from a STALE MAIN file.

        If `path` is omitted, the first discovered MAIN file is used. Missing
        data files, unreadable JSON, and unsupported payload shapes degrade to an
        empty list.
        """
        data_path = Path(path) if path is not None else self._first_main_file()
        if data_path is None:
            return []

        payload = self._read_json(data_path)
        raw_records = self._coerce_records(payload)
        return [
            self.normalize_record(record, sample_index=index, source_path=data_path)
            for index, record in enumerate(raw_records)
        ]

    def load_samples(self, path: Optional[Union[str, Path]] = None) -> list[dict[str, Any]]:
        """Alias for callers that use sample terminology."""
        return self.load_records(path)

    def normalize_record(
        self,
        record: dict[str, Any],
        *,
        sample_index: int = 0,
        source_path: Optional[Path] = None,
    ) -> dict[str, Any]:
        uid = record.get("uid") or record.get("sample_id") or record.get("id") or str(sample_index)
        sessions = record.get("haystack_session") or record.get("sessions") or []
        timestamps = record.get("timestamps") or []
        probing_queries = self._normalize_probing_queries(record)
        old_memory = record.get("M_old", record.get("old_info", ""))
        new_memory = record.get("M_new", record.get("new_info", ""))

        metadata_keys = (
            "time_gap",
            "relevant_session_index",
            "query_time",
            "attribute",
            "conflict_type",
        )
        metadata = {key: record.get(key) for key in metadata_keys if key in record}
        if source_path is not None:
            metadata["source_path"] = str(source_path)

        return {
            "sample_id": str(uid),
            "uid": uid,
            "sessions": sessions if isinstance(sessions, list) else [],
            "haystack_session": sessions if isinstance(sessions, list) else [],
            "timestamps": timestamps if isinstance(timestamps, list) else [],
            "query_time": record.get("query_time"),
            "probing_queries": probing_queries,
            "M_old": old_memory,
            "M_new": new_memory,
            "old_memory": old_memory,
            "new_memory": new_memory,
            "explanation": record.get("explanation", ""),
            "metadata": metadata,
            "raw": record,
        }

    def _first_main_file(self) -> Optional[Path]:
        files = self.discover_main_files()
        return files[0] if files else None

    def _read_json(self, path: Path) -> Any:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

    def _coerce_records(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("records", "samples", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [payload]
        return []

    def _normalize_probing_queries(self, record: dict[str, Any]) -> dict[str, str]:
        raw_queries = record.get("probing_queries")
        if not isinstance(raw_queries, dict):
            raw_queries = {}

        queries: dict[str, str] = {}
        for key in ("dim1_query", "dim2_query", "dim3_query"):
            value = raw_queries.get(key, record.get(key, ""))
            queries[key] = value if isinstance(value, str) else ""
        return queries

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


def write_jsonl(records: Iterable[object], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = records_to_jsonable(record)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            records.append(payload)
    return records


def records_to_jsonable(value: object) -> Any:
    if is_dataclass(value):
        return records_to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): records_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [records_to_jsonable(item) for item in value]
    return value

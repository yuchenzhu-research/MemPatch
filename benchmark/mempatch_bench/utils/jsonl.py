import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, List, Union, cast


def read_jsonl(path: Union[str, Path]) -> List[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Union[str, Path], records: List[Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for r in records:
            if is_dataclass(r) and not isinstance(r, type):
                data = asdict(cast(Any, r))
            elif isinstance(r, dict):
                data = r
            else:
                raise TypeError(f"Record must be dict or dataclass, got {type(r)}")
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GitHubSeed:
    pattern: str
    repo: str
    url_or_id: str
    raw_events: List[str] = field(default_factory=list)
    old_memory: str = ""
    candidate_new_memory: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_seeds_from_jsonl(path: Path) -> List[GitHubSeed]:
    seeds = []
    if not path.exists():
        return seeds
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            seeds.append(GitHubSeed(
                pattern=data["pattern"],
                repo=data["repo"],
                url_or_id=data["url_or_id"],
                raw_events=data.get("raw_events", []),
                old_memory=data.get("old_memory", ""),
                candidate_new_memory=data.get("candidate_new_memory", ""),
                metadata=data.get("metadata", {})
            ))
    return seeds

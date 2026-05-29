from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any


def get_git_commit_sha() -> str:
    """Safely retrieves the current git commit SHA."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def compute_file_sha256(filepath: str) -> str:
    """Computes the SHA256 checksum of a file if it exists."""
    if not os.path.exists(filepath):
        return ""
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


@dataclass
class RunConfiguration:
    """Frozen configuration details for a pipeline or evaluator run."""

    run_id: str
    git_commit_sha: str = field(default_factory=get_git_commit_sha)
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    stage_and_method_name: str = ""
    provider_name: str = ""
    model_id: str = ""
    model_revision_or_api_version: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    prompt_hashes: dict[str, str] = field(default_factory=dict)
    parser_schema_versions: dict[str, str] = field(default_factory=dict)
    cache_path: str = ""
    dataset_checksum: str = ""
    comparison_regime: str | None = None
    upstream_commit_sha: str | None = None
    scientific_status: str = "development_live"
    not_for_main_table: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunManifest:
    """Execution manifest wrapping configuration, execution metadata, costs, and outputs."""

    config: RunConfiguration
    aggregate_cost: dict[str, Any] = field(default_factory=dict)
    instance_count: int = 0
    output_checksum: str = ""
    output_path: str = ""
    errors_or_retries: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "aggregate_cost": self.aggregate_cost,
            "instance_count": self.instance_count,
            "output_checksum": self.output_checksum,
            "output_path": self.output_path,
            "errors_or_retries": self.errors_or_retries,
            "metadata": self.metadata,
        }

    def save(self, filepath: str) -> None:
        """Saves the run manifest to the given file path in structured JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        # Update output checksum if output file path is specified
        if self.output_path and not self.output_checksum:
            self.output_checksum = compute_file_sha256(self.output_path)
            
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

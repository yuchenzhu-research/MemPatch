"""Shared helpers for the real-seeded GitHub mining pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import yaml

GITHUB_API = "https://api.github.com"
GITHUB_WEB = "https://github.com"

MAINTAINER_ASSOCIATIONS = {"COLLABORATOR", "MEMBER", "OWNER"}

AUTHORITATIVE_EVENT_TYPES = {
    "maintainer_comment",
    "pr_merged",
    "release_note",
    "changelog",
    "docs",
}

PUBLIC_ALLOWED_TOP_LEVEL = {
    "schema_version",
    "scenario_id",
    "split",
    "domain",
    "workflow_context",
    "public_input",
    "tasks",
    "output_contract",
}

FORBIDDEN_PUBLIC_FIELDS = {
    "expected_answer",
    "expected_decision",
    "expected_evidence_event_ids",
    "expected_failure_diagnosis",
    "expected_followup_answer",
    "expected_followup_answer_key_facts",
    "expected_memory_operation",
    "expected_memory_state",
    "expected_memory_states",
    "failure_mode",
    "generation_metadata",
    "hidden_gold",
    "pattern",
    "primary_failure_mode",
    "resolver_trace",
    "source_pointers",
    "candidate_failure_modes",
    "candidate_memory_operations",
    "adjudication_notes",
}

TOKEN_PATTERNS = [
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}"),
]

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PRIVATE_URL_RE = re.compile(
    r"(?i)\b(?:https?://)?(?:localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+|[^/\s]+\.corp(?:\.[^/\s]+)?)(?:/[^\s]*)?"
)
SECURITY_DETAIL_RE = re.compile(
    r"(?i)\b("
    r"cve-\d{4}-\d{4,}|remote code execution|rce|sql injection|xss|csrf|ssrf|"
    r"privilege escalation|authentication bypass|deserialization exploit|proof[- ]of[- ]concept|"
    r"exploit chain|zero[- ]day|credential leak|private key|access token"
    r")\b"
)

FAILURE_BY_GROUP = {
    "deprecation": ["stale_memory_reuse", "under_update"],
    "release_state": ["stale_memory_reuse", "under_update"],
    "reversal": ["failure_to_release_or_restore", "conflict_collapse"],
    "scope_policy": ["scope_leakage", "policy_violation"],
    "triage": ["unnecessary_memory_write", "wrong_source_attribution"],
    "docs": ["wrong_source_attribution", "stale_memory_reuse"],
    "platform": ["scope_leakage", "over_update"],
}

OPERATIONS_BY_GROUP = {
    "deprecation": ["REVISE", "PRESERVE"],
    "release_state": ["REVISE", "MARK_UNRESOLVED"],
    "reversal": ["RESTORE_OR_RELEASE", "REVISE"],
    "scope_policy": ["RESTRICT_SCOPE", "BLOCK"],
    "triage": ["REJECT_NEW_MEMORY", "NO_WRITE"],
    "docs": ["REVISE", "MARK_UNRESOLVED"],
    "platform": ["RESTRICT_SCOPE", "MARK_UNRESOLVED"],
}

VALID_DECISIONS = {
    "use_current_memory",
    "escalate",
    "ask_clarification",
    "refuse_due_to_policy",
    "mark_unresolved",
}

VALID_OPERATIONS = {
    "PRESERVE",
    "REVISE",
    "RESTRICT_SCOPE",
    "BLOCK",
    "MARK_UNRESOLVED",
    "DELETE_OR_FORGET",
    "RESTORE_OR_RELEASE",
    "REJECT_NEW_MEMORY",
    "NO_WRITE",
    "ESCALATE",
}

VALID_STATUSES = {
    "current",
    "blocked",
    "unresolved",
    "out_of_scope",
    "should_not_store",
    "outdated",
    "deleted",
    "restored",
}

VALID_FAILURE_MODES = {
    "stale_memory_reuse",
    "under_update",
    "over_update",
    "conflict_collapse",
    "scope_leakage",
    "policy_violation",
    "wrong_source_attribution",
    "memory_hallucination",
    "unnecessary_memory_write",
    "failure_to_forget",
    "failure_to_release_or_restore",
}


class PipelineError(RuntimeError):
    """Raised for user-facing pipeline failures."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(value: str, *, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PipelineError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise PipelineError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            safe_row = sanitize_obj(row)
            handle.write(json.dumps(safe_row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_obj(row), ensure_ascii=False, sort_keys=True) + "\n")


def _truncate(value: str, max_chars: int) -> str:
    text = " ".join(value.replace("\r", "\n").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " [truncated]"


def sanitize_text(value: Any, *, max_chars: int = 1800) -> str:
    text = "" if value is None else str(value)
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = PRIVATE_URL_RE.sub("[REDACTED_PRIVATE_URL]", text)
    for pattern in TOKEN_PATTERNS:
        text = pattern.sub("[REDACTED_SECRET]", text)
    text = re.sub(r"(?i)(authorization:\s*bearer\s+)[A-Za-z0-9._-]+", r"\1[REDACTED_SECRET]", text)
    return _truncate(text, max_chars)


def sanitize_obj(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_obj(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_obj(item) for item in value]
    if isinstance(value, str):
        return sanitize_text(value, max_chars=5000)
    return value


def sensitive_findings(value: Any, prefix: str = "$") -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            findings.extend(sensitive_findings(item, f"{prefix}.{key}"))
        return findings
    if isinstance(value, list):
        for idx, item in enumerate(value):
            findings.extend(sensitive_findings(item, f"{prefix}[{idx}]"))
        return findings
    if not isinstance(value, str):
        return findings
    checks = (
        ("email", EMAIL_RE),
        ("token_or_secret", TOKEN_PATTERNS[0]),
        ("token_or_secret", TOKEN_PATTERNS[1]),
        ("token_or_secret", TOKEN_PATTERNS[2]),
        ("token_or_secret", TOKEN_PATTERNS[3]),
        ("secret_assignment", TOKEN_PATTERNS[4]),
        ("private_url", PRIVATE_URL_RE),
        ("security_detail", SECURITY_DETAIL_RE),
    )
    for kind, pattern in checks:
        if pattern.search(value):
            findings.append({"path": prefix, "kind": kind})
            break
    return findings


def has_sensitive_content(value: Any) -> bool:
    return bool(sensitive_findings(value))


def assert_token_not_in_command_context(token: str | None) -> None:
    """Fail before network requests if a token is visible in command arguments.

    We cannot inspect a user's terminal history from Python. This guard covers
    the command surfaces this process can see: argv and common Python process
    metadata. The pipeline never logs request headers or environment values.
    """
    if not token:
        return
    command_context = " ".join(sys.argv)
    command_context += "\n" + os.environ.get("PYTEST_CURRENT_TEST", "")
    if token and token in command_context:
        raise PipelineError("Refusing network request: token value appears in command context")


def github_token_from_env() -> str | None:
    token = os.environ.get("GITHUB_TOKEN")
    return token if token else None


def repo_slug_is_public_safe(repo: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo))


def public_github_url(url: Any) -> bool:
    if not isinstance(url, str) or not url.startswith("https://github.com/"):
        return False
    parsed = urlparse(url)
    return parsed.netloc.lower() == "github.com" and not PRIVATE_URL_RE.search(url)


def normalize_labels(raw_labels: Any) -> list[str]:
    labels: list[str] = []
    if isinstance(raw_labels, list):
        for label in raw_labels:
            if isinstance(label, dict):
                name = label.get("name")
            else:
                name = label
            if name:
                labels.append(sanitize_text(name, max_chars=120))
    return sorted(dict.fromkeys(labels))


def unique_preserve_order(values: Iterable[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        marker = json.dumps(value, sort_keys=True, default=str)
        if marker in seen:
            continue
        seen.add(marker)
        out.append(value)
    return out


def candidate_id(source_repo: str, source_url: str) -> str:
    return f"gh_{stable_hash(source_repo + '|' + source_url, length=20)}"


def event_id(candidate: str, source_type: str, ordinal: int, url: str = "") -> str:
    suffix = stable_hash(f"{candidate}|{source_type}|{ordinal}|{url}", length=8)
    return f"{candidate}_ev_{ordinal:02d}_{suffix}"


def terms_for_groups(query_groups: dict[str, list[str]]) -> list[str]:
    terms: list[str] = []
    for values in query_groups.values():
        terms.extend(str(value) for value in values)
    return sorted(dict.fromkeys(terms))


def infer_failure_modes(groups: Iterable[str], text: str = "") -> list[str]:
    modes: list[str] = []
    for group in groups:
        modes.extend(FAILURE_BY_GROUP.get(group, []))
    lowered = text.lower()
    if "duplicate" in lowered or "invalid" in lowered or "wontfix" in lowered:
        modes.extend(["unnecessary_memory_write", "wrong_source_attribution"])
    if "only windows" in lowered or "only linux" in lowered or "not supported" in lowered:
        modes.extend(["scope_leakage", "over_update"])
    if "reverted" in lowered or "rollback" in lowered:
        modes.extend(["failure_to_release_or_restore", "stale_memory_reuse"])
    if "permission" in lowered or "enterprise only" in lowered:
        modes.extend(["scope_leakage", "policy_violation"])
    return [mode for mode in unique_preserve_order(modes) if mode in VALID_FAILURE_MODES]


def infer_operations(groups: Iterable[str], text: str = "") -> list[str]:
    operations: list[str] = []
    for group in groups:
        operations.extend(OPERATIONS_BY_GROUP.get(group, []))
    lowered = text.lower()
    if "duplicate" in lowered or "invalid" in lowered or "wontfix" in lowered:
        operations.extend(["REJECT_NEW_MEMORY", "NO_WRITE"])
    if "only windows" in lowered or "only linux" in lowered or "not supported" in lowered:
        operations.extend(["RESTRICT_SCOPE", "MARK_UNRESOLVED"])
    if "reverted" in lowered or "rollback" in lowered:
        operations.extend(["RESTORE_OR_RELEASE", "REVISE"])
    if "permission" in lowered or "enterprise only" in lowered:
        operations.extend(["RESTRICT_SCOPE", "BLOCK"])
    return [op for op in unique_preserve_order(operations) if op in VALID_OPERATIONS]


def summarize_counter(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = row.get(key)
        if isinstance(value, list):
            counter.update(str(item) for item in value)
        elif value is not None:
            counter[str(value)] += 1
    return dict(sorted(counter.items()))

#!/usr/bin/env python
"""Mine public GitHub evidence into provisional real-seeded candidates."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from http.client import HTTPException
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.real_seeded.common import (
    AUTHORITATIVE_EVENT_TYPES,
    GITHUB_API,
    MAINTAINER_ASSOCIATIONS,
    PipelineError,
    append_jsonl,
    assert_token_not_in_command_context,
    candidate_id,
    event_id,
    github_token_from_env,
    infer_failure_modes,
    infer_operations,
    normalize_labels,
    public_github_url,
    read_jsonl,
    read_yaml,
    repo_slug_is_public_safe,
    sanitize_obj,
    sanitize_text,
    sensitive_findings,
    terms_for_groups,
    utc_now,
    write_jsonl,
)

DOC_PATHS = (
    "CHANGELOG.md",
    "CHANGES.md",
    "RELEASE.md",
    "RELEASES.md",
    "docs/changelog.md",
    "docs/CHANGELOG.md",
    "docs/release-notes.md",
    "docs/releases.md",
)


class RateLimitError(PipelineError):
    """Raised when GitHub rate limits prevent further mining."""


class GitHubClient:
    def __init__(self, *, cache_dir: Path | None = None, timeout: int = 30, max_rate_wait: int = 60) -> None:
        self.token = github_token_from_env()
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_rate_wait = max_rate_wait
        self.network_requests = 0
        self.cache_hits = 0
        self.last_rate: dict[str, Any] = {}
        self.failure_counts: Counter[str] = Counter()
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, url: str) -> Path | None:
        if not self.cache_dir:
            return None
        from tools.real_seeded.common import stable_hash

        return self.cache_dir / f"{stable_hash(url, length=32)}.json"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "MemPatch-Bench-real-seeded-miner",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        return headers

    def get(self, url: str, *, use_cache: bool = True) -> Any:
        assert_token_not_in_command_context(self.token)
        cache_path = self._cache_path(url)
        if use_cache and cache_path and cache_path.exists():
            self.cache_hits += 1
            return json.loads(cache_path.read_text(encoding="utf-8"))
        data = None
        for attempt in range(2):
            request = Request(url, headers=self._headers())
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = response.read().decode("utf-8")
                    data = json.loads(payload) if payload else None
                    self.network_requests += 1
                    self._remember_rate(response.headers)
                    break
            except HTTPError as exc:
                self.failure_counts[f"http_{exc.code}"] += 1
                self._remember_rate(exc.headers)
                if exc.code in {403, 429} and attempt == 0 and self._maybe_wait_for_rate_limit(exc):
                    continue
                if exc.code in {403, 429}:
                    self._raise_rate_limit(exc)
                message = self._safe_error_message(exc)
                raise PipelineError(f"GitHub request failed for {self._safe_url(url)}: {message}") from exc
            except (HTTPException, TimeoutError, ConnectionError) as exc:
                self.failure_counts[type(exc).__name__] += 1
                if attempt == 0:
                    time.sleep(1)
                    continue
                raise PipelineError(f"GitHub connection failed for {self._safe_url(url)}: {type(exc).__name__}") from exc
            except URLError as exc:
                self.failure_counts["url_error"] += 1
                raise PipelineError(f"GitHub is unreachable for {self._safe_url(url)}: {exc.reason}") from exc
        if cache_path:
            cache_path.write_text(
                json.dumps(sanitize_obj(data), ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
        return data

    def _remember_rate(self, headers: Any) -> None:
        if not headers:
            return
        self.last_rate = {
            "limit": headers.get("X-RateLimit-Limit"),
            "remaining": headers.get("X-RateLimit-Remaining"),
            "reset": headers.get("X-RateLimit-Reset"),
            "resource": headers.get("X-RateLimit-Resource"),
        }

    def _maybe_wait_for_rate_limit(self, exc: HTTPError) -> bool:
        remaining = (exc.headers or {}).get("X-RateLimit-Remaining")
        reset = (exc.headers or {}).get("X-RateLimit-Reset")
        if remaining == "0" and reset:
            wait = max(0, int(reset) - int(time.time()) + 1)
            if wait <= self.max_rate_wait:
                print(f"GitHub rate limit reached; waiting {wait}s before retrying.")
                time.sleep(wait)
                return True
        return False

    def _raise_rate_limit(self, exc: HTTPError) -> None:
        remaining = (exc.headers or {}).get("X-RateLimit-Remaining")
        reset = (exc.headers or {}).get("X-RateLimit-Reset")
        if remaining == "0" and reset:
            wait = max(0, int(reset) - int(time.time()) + 1)
            self.failure_counts["rate_limit"] += 1
            raise RateLimitError(
                f"GitHub rate limit reached; reset is {wait}s away, above --rate-limit-wait-seconds={self.max_rate_wait}"
            )
        self.failure_counts["abuse_or_secondary_rate_limit"] += 1
        raise RateLimitError("GitHub returned a rate-limit or abuse-limit response")

    def _safe_url(self, url: str) -> str:
        return sanitize_text(url, max_chars=240)

    def _safe_error_message(self, exc: HTTPError) -> str:
        try:
            payload = exc.read().decode("utf-8")
        except Exception:
            payload = str(exc)
        return sanitize_text(payload, max_chars=500)

    def diagnostic(self) -> dict[str, Any]:
        mode = "authenticated" if self.token else "unauthenticated"
        if not self.token:
            print("GITHUB_TOKEN is not set; running unauthenticated and subject to lower GitHub rate limits.")
        else:
            print("GITHUB_TOKEN is set; using authenticated GitHub requests without printing the token.")
        data = self.get(f"{GITHUB_API}/rate_limit", use_cache=False)
        core = ((data or {}).get("resources") or {}).get("core") or {}
        search = ((data or {}).get("resources") or {}).get("search") or {}
        report = {
            "mode": mode,
            "core_remaining": core.get("remaining"),
            "core_limit": core.get("limit"),
            "search_remaining": search.get("remaining"),
            "search_limit": search.get("limit"),
        }
        print(
            "GitHub connectivity ok: "
            f"{mode}, core {report['core_remaining']}/{report['core_limit']}, "
            f"search {report['search_remaining']}/{report['search_limit']}."
        )
        return report


def issue_search_url(repo: str, query: str, item_type: str, page: int, per_page: int) -> str:
    q = f"repo:{repo} {query} type:{item_type}"
    params = urlencode({"q": q, "sort": "updated", "order": "desc", "per_page": per_page, "page": page})
    return f"{GITHUB_API}/search/issues?{params}"


def group_query(terms: list[str]) -> str:
    quoted = []
    for term in terms:
        text = str(term).strip()
        if not text:
            continue
        if " " in text:
            quoted.append(f'"{text}"')
        else:
            quoted.append(text)
    if len(quoted) == 1:
        return quoted[0]
    return "(" + " OR ".join(quoted) + ")"


def api_repo_url(repo: str, suffix: str = "") -> str:
    owner, name = repo.split("/", 1)
    suffix = suffix.lstrip("/")
    return f"{GITHUB_API}/repos/{quote(owner)}/{quote(name)}" + (f"/{suffix}" if suffix else "")


def safe_event_text(*parts: Any) -> str:
    joined = "\n\n".join(str(part) for part in parts if part)
    return sanitize_text(joined, max_chars=1800)


def is_public_repo(client: GitHubClient, repo: str) -> bool:
    if not repo_slug_is_public_safe(repo):
        return False
    data = client.get(api_repo_url(repo))
    return not bool((data or {}).get("private"))


def event_from_issue(candidate: str, repo: str, issue: dict[str, Any], ordinal: int) -> dict[str, Any]:
    is_pr = "pull_request" in issue
    source_type = "pr_body" if is_pr else "issue_body"
    url = issue.get("html_url") or ""
    return {
        "event_id": event_id(candidate, source_type, ordinal, url),
        "timestamp": issue.get("created_at") or issue.get("updated_at"),
        "source_type": source_type,
        "url": url,
        "text": safe_event_text(issue.get("title"), issue.get("body")),
    }


def event_from_comment(candidate: str, comment: dict[str, Any], ordinal: int) -> dict[str, Any]:
    association = str(comment.get("author_association") or "").upper()
    source_type = "maintainer_comment" if association in MAINTAINER_ASSOCIATIONS else "issue_comment"
    url = comment.get("html_url") or ""
    return {
        "event_id": event_id(candidate, source_type, ordinal, url),
        "timestamp": comment.get("created_at") or comment.get("updated_at"),
        "source_type": source_type,
        "url": url,
        "text": safe_event_text(comment.get("body")),
    }


def event_from_pr(candidate: str, pr: dict[str, Any], ordinal: int) -> dict[str, Any] | None:
    if not pr or not pr.get("merged_at"):
        return None
    url = pr.get("html_url") or ""
    return {
        "event_id": event_id(candidate, "pr_merged", ordinal, url),
        "timestamp": pr.get("merged_at"),
        "source_type": "pr_merged",
        "url": url,
        "text": safe_event_text("Merged PR", pr.get("title"), pr.get("body")),
    }


def _title_tokens(title: Any) -> set[str]:
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "type",
        "support",
        "issue",
        "feature",
        "request",
        "fix",
        "fixed",
        "update",
        "change",
        "changes",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{3,}", str(title or "").lower())
        if token not in stop
    }


def release_matches(
    release: dict[str, Any],
    issue_number: int | None,
    pr_numbers: list[int],
    title: Any = "",
) -> bool:
    text = f"{release.get('name') or ''}\n{release.get('tag_name') or ''}\n{release.get('body') or ''}".lower()
    if issue_number and (f"#{issue_number}" in text or f"/issues/{issue_number}" in text):
        return True
    for pr in pr_numbers:
        if f"#{pr}" in text or f"/pull/{pr}" in text:
            return True
    tokens = _title_tokens(title)
    return len(tokens & set(re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{3,}", text))) >= 2


def event_from_release(candidate: str, release: dict[str, Any], ordinal: int) -> dict[str, Any]:
    url = release.get("html_url") or ""
    return {
        "event_id": event_id(candidate, "release_note", ordinal, url),
        "timestamp": release.get("published_at") or release.get("created_at"),
        "source_type": "release_note",
        "url": url,
        "text": safe_event_text(release.get("name") or release.get("tag_name"), release.get("body")),
    }


def fetch_paginated(client: GitHubClient, url_builder: Any, *, max_pages: int) -> list[Any]:
    items: list[Any] = []
    for page in range(1, max_pages + 1):
        data = client.get(url_builder(page))
        if isinstance(data, dict) and "items" in data:
            page_items = data.get("items") or []
        elif isinstance(data, list):
            page_items = data
        else:
            page_items = []
        items.extend(page_items)
        if not page_items:
            break
    return items


def fetch_issue_comments(client: GitHubClient, repo: str, number: int, *, max_pages: int = 1) -> list[dict[str, Any]]:
    try:
        return fetch_paginated(
            client,
            lambda page: api_repo_url(repo, f"issues/{number}/comments?per_page=100&page={page}"),
            max_pages=max_pages,
        )
    except RateLimitError:
        raise
    except PipelineError:
        return []


def fetch_pr(client: GitHubClient, repo: str, number: int) -> dict[str, Any] | None:
    try:
        data = client.get(api_repo_url(repo, f"pulls/{number}"))
    except PipelineError:
        return None
    return data if isinstance(data, dict) else None


def fetch_releases(client: GitHubClient, repo: str, *, max_pages: int = 1) -> list[dict[str, Any]]:
    try:
        return fetch_paginated(
            client,
            lambda page: api_repo_url(repo, f"releases?per_page=30&page={page}"),
            max_pages=max_pages,
        )
    except PipelineError:
        return []


def linked_pr_numbers(text: str, repo: str) -> list[int]:
    numbers: list[int] = []
    for match in re.finditer(r"(?:pull/|#)(\d{1,7})", text):
        numbers.append(int(match.group(1)))
    repo_pattern = re.escape(f"https://github.com/{repo}/pull/")
    for match in re.finditer(repo_pattern + r"(\d{1,7})", text):
        numbers.append(int(match.group(1)))
    return sorted(set(numbers))


def fetch_doc_snippets(client: GitHubClient, repo: str, terms: list[str]) -> list[dict[str, Any]]:
    snippets: list[dict[str, Any]] = []
    for path in DOC_PATHS:
        try:
            data = client.get(api_repo_url(repo, f"contents/{quote(path)}"))
        except PipelineError:
            continue
        if not isinstance(data, dict) or data.get("type") != "file":
            continue
        if int(data.get("size") or 0) > 500_000:
            continue
        content = data.get("content")
        if not content:
            continue
        try:
            text = base64.b64decode(content).decode("utf-8", errors="replace")
        except Exception:
            continue
        lowered = text.lower()
        matched = [term for term in terms if term.lower() in lowered]
        if not matched:
            continue
        lines = text.splitlines()
        selected: list[str] = []
        for idx, line in enumerate(lines):
            if any(term.lower() in line.lower() for term in matched):
                start = max(0, idx - 1)
                end = min(len(lines), idx + 2)
                selected.extend(lines[start:end])
                if len(selected) > 12:
                    break
        snippets.append(
            {
                "path": path,
                "html_url": data.get("html_url") or f"https://github.com/{repo}/blob/HEAD/{path}",
                "text": sanitize_text("\n".join(selected), max_chars=1200),
                "terms": matched,
            }
        )
    return snippets


def event_from_doc(candidate: str, doc: dict[str, Any], ordinal: int) -> dict[str, Any]:
    url = doc.get("html_url") or ""
    source_type = "changelog" if "change" in str(doc.get("path", "")).lower() else "docs"
    return {
        "event_id": event_id(candidate, source_type, ordinal, url),
        "timestamp": None,
        "source_type": source_type,
        "url": url,
        "text": safe_event_text(f"{doc.get('path')} snippet", doc.get("text")),
    }


def doc_matches(doc: dict[str, Any], issue: dict[str, Any], pr_numbers: list[int]) -> bool:
    text = f"{doc.get('path') or ''}\n{doc.get('text') or ''}".lower()
    issue_number = int(issue.get("number") or 0)
    if issue_number and (f"#{issue_number}" in text or f"/issues/{issue_number}" in text):
        return True
    for pr in pr_numbers:
        if f"#{pr}" in text or f"/pull/{pr}" in text:
            return True
    tokens = _title_tokens(issue.get("title"))
    return len(tokens & set(re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{3,}", text))) >= 2


def build_candidate(
    *,
    repo: str,
    issue: dict[str, Any],
    query_group: str,
    query_terms: list[str],
    comments: list[dict[str, Any]] | None = None,
    prs: list[dict[str, Any]] | None = None,
    releases: list[dict[str, Any]] | None = None,
    docs: list[dict[str, Any]] | None = None,
    snapshot_time: str | None = None,
) -> dict[str, Any] | None:
    source_url = issue.get("html_url") or ""
    if not public_github_url(source_url):
        return None
    cid = candidate_id(repo, source_url)
    comments = comments or []
    prs = prs or []
    releases = releases or []
    docs = docs or []
    raw_events: list[dict[str, Any]] = []
    raw_events.append(event_from_issue(cid, repo, issue, 1))
    ordinal = 2
    for comment in comments[:20]:
        ev = event_from_comment(cid, comment, ordinal)
        if ev["text"]:
            raw_events.append(ev)
            ordinal += 1
    merged_prs: list[int] = []
    for pr in prs:
        ev = event_from_pr(cid, pr, ordinal)
        if ev:
            raw_events.append(ev)
            merged_prs.append(int(pr.get("number") or 0))
            ordinal += 1
    issue_number = int(issue.get("number") or 0)
    pr_numbers = sorted({n for n in linked_pr_numbers(json.dumps(issue) + " " + json.dumps(comments), repo) if n})
    pr_numbers = sorted(set(pr_numbers + merged_prs))
    for release in releases:
        if release_matches(release, issue_number, pr_numbers, issue.get("title")):
            raw_events.append(event_from_release(cid, release, ordinal))
            ordinal += 1
            if sum(1 for ev in raw_events if ev["source_type"] == "release_note") >= 2:
                break
    for doc in docs:
        if doc_matches(doc, issue, pr_numbers):
            raw_events.append(event_from_doc(cid, doc, ordinal))
            ordinal += 1
            if sum(1 for ev in raw_events if ev["source_type"] in {"docs", "changelog"}) >= 2:
                break
    raw_events = [
        ev
        for ev in raw_events
        if ev.get("text") and public_github_url(ev.get("url", "")) and not sensitive_findings(ev)
    ]
    if not raw_events:
        return None
    text_blob = " ".join([str(issue.get("title") or ""), *(ev.get("text", "") for ev in raw_events)])
    groups = [query_group]
    candidate = {
        "candidate_id": cid,
        "source_repo": repo,
        "source_urls": sorted({source_url, *(ev.get("url") for ev in raw_events if ev.get("url"))}),
        "issue_number": issue_number,
        "pr_numbers": pr_numbers,
        "release_urls": sorted(
            {ev["url"] for ev in raw_events if ev.get("source_type") == "release_note" and ev.get("url")}
        ),
        "doc_urls": sorted(
            {ev["url"] for ev in raw_events if ev.get("source_type") in {"docs", "changelog"} and ev.get("url")}
        ),
        "title": sanitize_text(issue.get("title"), max_chars=300),
        "labels": normalize_labels(issue.get("labels") or []),
        "state": sanitize_text(issue.get("state"), max_chars=40),
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "closed_at": issue.get("closed_at"),
        "raw_events": raw_events,
        "candidate_failure_modes": infer_failure_modes(groups, text_blob)[:2],
        "candidate_memory_operations": infer_operations(groups, text_blob)[:2],
        "snapshot_time": snapshot_time or utc_now(),
        "license_provenance_note": "Public GitHub metadata and user-authored issue/PR/release/doc text; preserve source URLs and repository licenses/terms.",
        "retrieval_query_terms": {"group": query_group, "terms": query_terms},
        "provisional": not any(ev["source_type"] in AUTHORITATIVE_EVENT_TYPES for ev in raw_events),
    }
    if sensitive_findings(candidate):
        return None
    return sanitize_obj(candidate)


def selected_repo_rows(repos: list[str], client: GitHubClient) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repo in repos:
        try:
            data = client.get(api_repo_url(repo))
        except PipelineError as exc:
            rows.append({"repo": repo, "selected": False, "reason": str(exc)})
            continue
        rows.append(
            {
                "repo": repo,
                "selected": not bool(data.get("private")),
                "private": bool(data.get("private")),
                "html_url": data.get("html_url"),
                "license": (data.get("license") or {}).get("spdx_id"),
                "default_branch": data.get("default_branch"),
            }
        )
    return rows


def mine(args: argparse.Namespace) -> dict[str, Any]:
    repos_config = read_yaml(args.repos)
    query_groups = read_yaml(args.queries)
    repos = list(repos_config.get("tier_a") or []) + list(repos_config.get("tier_b") or [])
    if args.repo_limit:
        repos = repos[: args.repo_limit]
    if args.query_groups:
        allowed = set(args.query_groups.split(","))
        query_groups = {key: value for key, value in query_groups.items() if key in allowed}
    client = GitHubClient(cache_dir=args.cache, max_rate_wait=args.rate_limit_wait_seconds)
    diagnostic = client.diagnostic()
    if args.diagnostic_only:
        return {"diagnostic": diagnostic}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.touch(exist_ok=True)
    existing = read_jsonl(args.out)
    seen_urls = {((row.get("source_repo"), row.get("source_urls", [""])[0])) for row in existing}
    seen_ids = {row.get("candidate_id") for row in existing}
    selected_path = args.selected_repos or args.out.parent / "selected_repos.json"
    if selected_path.exists():
        selected_rows = json.loads(selected_path.read_text(encoding="utf-8"))
    else:
        selected_rows = selected_repo_rows(repos, client)
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_path.write_text(json.dumps(selected_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    public_repos = [row["repo"] for row in selected_rows if row.get("selected")]
    snapshot = utc_now()
    new_count = 0
    stop_reason = "completed"
    try:
        for repo in public_repos:
            if len(existing) + new_count >= args.target_raw:
                break
            all_terms = terms_for_groups(query_groups)
            releases = fetch_releases(client, repo, max_pages=args.release_pages)
            docs = fetch_doc_snippets(client, repo, all_terms) if args.include_docs else []
            for group, terms in query_groups.items():
                if len(existing) + new_count >= args.target_raw:
                    break
                query = group_query([str(term) for term in terms])
                for item_type in ("issue", "pr"):
                    if len(existing) + new_count >= args.target_raw:
                        break
                    try:
                        items = fetch_paginated(
                            client,
                            lambda page, repo=repo, query=query, item_type=item_type: issue_search_url(
                                repo, query, item_type, page, args.per_page
                            ),
                            max_pages=args.search_pages,
                        )
                    except RateLimitError:
                        raise
                    except PipelineError as exc:
                        client.failure_counts["search_failed"] += 1
                        print(
                            "Skipping failed search "
                            f"repo={repo} group={group} type={item_type}: "
                            f"{sanitize_text(str(exc), max_chars=240)}"
                        )
                        continue
                    for issue in items[: args.max_items_per_query]:
                        source_url = issue.get("html_url") or ""
                        key = (repo, source_url)
                        if key in seen_urls:
                            continue
                        number = int(issue.get("number") or 0)
                        comments = fetch_issue_comments(client, repo, number, max_pages=args.comment_pages)
                        text_for_links = json.dumps(issue) + " " + json.dumps(comments)
                        pr_numbers = linked_pr_numbers(text_for_links, repo)
                        if "pull_request" in issue and number not in pr_numbers:
                            pr_numbers.append(number)
                        prs = [pr for n in pr_numbers[:4] if (pr := fetch_pr(client, repo, n))]
                        candidate = build_candidate(
                            repo=repo,
                            issue=issue,
                            query_group=group,
                            query_terms=list(terms),
                            comments=comments,
                            prs=prs,
                            releases=releases,
                            docs=docs,
                            snapshot_time=snapshot,
                        )
                        if not candidate or candidate["candidate_id"] in seen_ids:
                            continue
                        append_jsonl(args.out, candidate)
                        seen_urls.add(key)
                        seen_ids.add(candidate["candidate_id"])
                        new_count += 1
                        if (len(existing) + new_count) % 25 == 0:
                            print(
                                f"Collected {len(existing) + new_count} raw candidates "
                                f"({new_count} new); last repo={repo}, group={group}."
                            )
                        if len(existing) + new_count >= args.target_raw:
                            break
    except RateLimitError as exc:
        stop_reason = str(exc)
    except PipelineError as exc:
        stop_reason = str(exc)
    total = len(existing) + new_count
    report = {
        "raw_candidates_total": total,
        "new_candidates": new_count,
        "target_raw": args.target_raw,
        "target_reached": total >= args.target_raw,
        "stop_reason": stop_reason,
        "network_requests": client.network_requests,
        "cache_hits": client.cache_hits,
        "last_rate": client.last_rate,
        "api_failures": dict(sorted(client.failure_counts.items())),
        "selected_repos": len(public_repos),
    }
    report_path = args.report_out or args.out.parent / "mining_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(sanitize_obj(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if total < args.target_raw:
        print("Raw target not reached; rerun with GITHUB_TOKEN set or a warmer cache to continue from the JSONL output.")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("datasets/v1.4_real_seeded/cache"))
    parser.add_argument("--selected-repos", type=Path)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--target-raw", type=int, default=300)
    parser.add_argument("--repo-limit", type=int)
    parser.add_argument("--query-groups", help="Comma-separated query group names for smoke tests.")
    parser.add_argument("--search-pages", type=int, default=1)
    parser.add_argument("--comment-pages", type=int, default=1)
    parser.add_argument("--release-pages", type=int, default=1)
    parser.add_argument("--per-page", type=int, default=30)
    parser.add_argument("--max-items-per-query", type=int, default=8)
    parser.add_argument("--rate-limit-wait-seconds", type=int, default=60)
    parser.add_argument("--include-docs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostic-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        mine(args)
    except PipelineError as exc:
        print(f"Mining stopped cleanly: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

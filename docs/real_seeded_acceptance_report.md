# Real-Seeded Acceptance Report

Status: pipeline implemented and smoke-tested locally. The full 300/80 mining target was not reached in this environment because `GITHUB_TOKEN` was not set and unauthenticated GitHub requests hit connection/rate-limit failures.

## Repositories Inspected

Configured Tier A repos: pydantic/pydantic, fastapi/fastapi, langchain-ai/langchain, run-llama/llama_index, pandas-dev/pandas, dbt-labs/dbt-core, grafana/grafana, apache/superset, ray-project/ray, pytest-dev/pytest.

Configured Tier B fallback repos: django/django, pallets/flask, scikit-learn/scikit-learn, metabase/metabase, apache/airflow, kubernetes/kubernetes, microsoft/vscode.

The unauthenticated full run wrote `data/v1.4_real_seeded/raw/selected_repos.json` with 17 configured repos inspected and 13 confirmed selected before rate/network failures. Four repo metadata checks did not complete: ray-project/ray, pytest-dev/pytest, django/django, and microsoft/vscode.

## Full Run Counts

- Raw candidates collected: 0
- Accepted candidates: 0
- Rejected candidates: 0
- Top rejection reasons: none, because no raw candidates were collected
- Per-repo distribution: none
- Per-failure-mode distribution: none
- Per-operation distribution: none
- Average raw events per accepted candidate: 0
- Candidates with release/changelog/docs evidence: 0
- Candidates with merged PR evidence: 0
- Target 300 raw / 80 accepted reached: no

## Smoke Run Counts

Smoke command: 2 repos, query groups `deprecation,release_state`, target 10.

- Raw candidates collected: 4
- Accepted candidates: 2
- Rejected candidates: 2
- Top rejection reasons: duplicate source candidate; needs subjective interpretation
- Per-repo distribution: pydantic/pydantic only
- Per-failure-mode distribution: stale_memory_reuse only
- Per-operation distribution: REVISE only
- Smoke audit result: passed after duplicate-source audit was scoped to original candidate IDs rather than shared release-note URLs

## Risks And Next Steps

- Export `GITHUB_TOKEN` in the shell and rerun the full command. The pasted chat credential was intentionally not used.
- Review accepted candidates manually before treating labels as final truth; the filter is conservative but still heuristic.
- Re-run the smoke output after authenticated mining if query/release matching changes, because stale cached smoke candidates may reflect earlier heuristics.
- Keep security-sensitive issues out of outputs. The miner/filter/audit redact or reject emails, token-like strings, private URLs, and sensitive exploit-detail patterns.

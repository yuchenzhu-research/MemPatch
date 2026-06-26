# Real-Seeded Local Runbook

This pipeline mines only public GitHub repositories and reads authentication only from `GITHUB_TOKEN`.
Do not paste tokens into commands, config files, logs, notebooks, or JSON outputs.

## Safety Diagnostic

```bash
export GITHUB_TOKEN="..."
python3 scripts/real_seeded/mine_github_candidates.py \
  --repos configs/real_seeded/repos_tier_a.yaml \
  --queries configs/real_seeded/query_terms.yaml \
  --out data/v1.4_real_seeded/raw/github_candidates.jsonl \
  --cache data/v1.4_real_seeded/cache \
  --diagnostic-only
```

If `GITHUB_TOKEN` is unset, the miner runs unauthenticated and warns about rate limits. It never prints the token.

## Full Pipeline

Do not normalize immediately after mining. First generate and inspect the mining quality report.

```bash
export GITHUB_TOKEN="..."
python3 scripts/real_seeded/mine_github_candidates.py \
  --repos configs/real_seeded/repos_tier_a.yaml \
  --queries configs/real_seeded/query_terms.yaml \
  --out data/v1.4_real_seeded/raw/github_candidates.jsonl \
  --cache data/v1.4_real_seeded/cache \
  --target-raw 300 \
  --report-out data/v1.4_real_seeded/raw/mining_report.json

python3 scripts/real_seeded/report_mining_quality.py \
  --raw data/v1.4_real_seeded/raw/github_candidates.jsonl \
  --mining-summary data/v1.4_real_seeded/raw/mining_report.json \
  --out data/v1.4_real_seeded/raw/mining_quality_report.md \
  --json-out data/v1.4_real_seeded/raw/mining_quality_report.json
```

Continue mining if the report does not satisfy the raw gate:

- Raw candidates at least 300
- Average raw events at least 3
- At least 100 candidates with maintainer, merged PR, release, changelog, or docs evidence
- Open-only candidates are not dominant
- Duplicate sources are not dominant

Only after the raw mining gate passes, run filtering:

```bash

python3 scripts/real_seeded/filter_candidates.py \
  --in data/v1.4_real_seeded/raw/github_candidates.jsonl \
  --accepted data/v1.4_real_seeded/filtered/accepted_candidates.jsonl \
  --rejected data/v1.4_real_seeded/filtered/rejected_candidates.jsonl
```

Inspect accepted candidates before normalization. The accepted gate is:

- At least 80 high-quality accepted candidates
- Average accepted evidence events at least 3
- At least 6 failure modes
- At least 4 memory operations
- No one-event issue-body-only accepted cases
- No obvious PII, secret, private URL, or sensitive security leakage

Only after the accepted quality gate passes, normalize and audit:

```bash

python3 scripts/real_seeded/normalize_real_seeded.py \
  --accepted data/v1.4_real_seeded/filtered/accepted_candidates.jsonl \
  --public data/v1.4/public/real_seeded_challenge.jsonl \
  --labels data/v1.4/labels/real_seeded_challenge.labels.jsonl

python3 scripts/real_seeded/audit_real_seeded.py \
  --public data/v1.4/public/real_seeded_challenge.jsonl \
  --labels data/v1.4/labels/real_seeded_challenge.labels.jsonl \
  --out data/v1.4/audits/audit_report_real_seeded.json
```

## Smoke Test

```bash
python3 scripts/real_seeded/mine_github_candidates.py \
  --repos configs/real_seeded/repos_tier_a.yaml \
  --queries configs/real_seeded/query_terms.yaml \
  --out data/v1.4_real_seeded/raw/github_candidates.smoke.jsonl \
  --cache data/v1.4_real_seeded/cache_smoke \
  --repo-limit 2 \
  --query-groups deprecation,release_state \
  --target-raw 10 \
  --max-items-per-query 2
```

Then run the same filter, normalize, and audit commands against the smoke paths.

## Output Policy

Public rows must not contain label-only fields such as `expected_*`, `failure_mode`, `pattern`, `resolver_trace`, `source_pointers`, or adjudication notes. Private labels keep the expected memory operation, expected evidence IDs, source pointers, evidence span hashes, provenance notes, and original candidate IDs.

Rejected candidates are retained only after sanitization. Any row containing emails, token-like strings, private URLs, or sensitive exploit details must fail the filter or audit.

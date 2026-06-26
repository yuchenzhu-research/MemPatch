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

```bash
export GITHUB_TOKEN="..."
python3 scripts/real_seeded/mine_github_candidates.py \
  --repos configs/real_seeded/repos_tier_a.yaml \
  --queries configs/real_seeded/query_terms.yaml \
  --out data/v1.4_real_seeded/raw/github_candidates.jsonl \
  --cache data/v1.4_real_seeded/cache \
  --target-raw 300

python3 scripts/real_seeded/filter_candidates.py \
  --in data/v1.4_real_seeded/raw/github_candidates.jsonl \
  --accepted data/v1.4_real_seeded/filtered/accepted_candidates.jsonl \
  --rejected data/v1.4_real_seeded/filtered/rejected_candidates.jsonl

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

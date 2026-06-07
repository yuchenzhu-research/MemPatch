# Decision boundary audit

Use `scripts/audit_decision_boundary.py` before training or releasing MemPatch scenario JSONL.

## Quick start

```bash
python scripts/audit_decision_boundary.py \
  --data local/MemPatch/train/scenarios.jsonl \
  --data local/MemPatch/main/scenarios.jsonl \
  --data local/MemPatch/hard/scenarios.jsonl \
  --out-json local/results/decision_boundary_audit.json \
  --out-md local/results/decision_boundary_audit.md
```

Add `--no-fail` to always exit 0 (for overnight pipelines). Without it, the script exits **2** on release-blocking failures:

- `ask_clarification` and `escalate` share a `core_event_signature`
- ask / escalate / mark share `core_event_signature` or full sanitized public view
- non-answer decisions missing visible trigger coverage in `public_input`
- `gold_public_consistency` violations (answer text vs public triggers)

## What it checks

- per-split `expected_decision` histogram
- `core_event_signature` from the first 2–3 non-background `event_trace` rows
- full sanitized public-view hash collisions
- cross-decision signature collisions (especially ask / escalate / mark_unresolved)
- ask vs escalate public event-text bigram Jaccard (warn threshold default 0.35)
- `pattern` / `pattern_trap_type` / `decision_variant` / `decision_triggers` cross tables
- `mark_unresolved` ci-derived vs non-ci breakdown
- global `pattern × decision × split` matrix in JSON summary

## Overnight helper (local only)

```bash
bash local/scripts/run_overnight_mempatch.sh
```

Set `SKIP_V4_TRAIN=1` to audit + eval only without the rank-16 ablation run.

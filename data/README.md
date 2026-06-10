# Tracked data artifacts

| Path | Purpose |
|------|---------|
| `mempatch/boundary_audit_v13.json` | Release gate audit report for v1.3 full split (4000 rows) |

Scenario JSONL is generated locally:

```bash
python scripts/data/generate_mempatch.py --full --out-dir local/data/mempatch
```

The bundle is gitignored under `local/`; publish separately on a dataset host when ready.

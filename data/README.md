# Tracked data artifacts

| Path | Purpose |
|------|---------|
| `mempatch/boundary_audit_v13.json` | Release gate audit for v1.3 `train` (3500) + `test` (500) |

Scenario JSONL is generated locally:

```bash
python scripts/data/generate_mempatch.py --full --out-dir local/data/mempatch
```

The bundle is gitignored under `local/`; publish separately on a dataset host when ready.

HF release layout is **train/** + **test/** only. Remove any legacy Hub directories (`main/`, `hard/`, `validation/`, `v1_2_preview/`) before publishing.

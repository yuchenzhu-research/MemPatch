# Reproducibility

## Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The deterministic core and all offline modes use only the standard library plus
the packaged dependencies. Live API runs additionally read credentials from a
`.env` file in the repo root (e.g. `SILICONFLOW_API_KEY=...`).

## Verify the build

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests experiments
python3 -m pytest
```

## Offline smoke (no credentials)

```bash
python3 scripts/evaluate.py stage-a --mock  --max-cases 2 --output-dir outputs/runs/smoke_a
python3 scripts/evaluate.py stage-b --mock  --max-cases 2 --output-dir outputs/runs/smoke_b
python3 scripts/evaluate.py stage-c --smoke --max-cases 3 --output-dir outputs/runs/smoke_c
```

## Run directory contents

Each run writes a self-describing directory under `outputs/` (git-ignored):

| File | Contents |
| --- | --- |
| `stage_a_raw.jsonl` / `stage_b_raw.jsonl` | raw proposer / judge responses |
| `stage_a_parsed.jsonl` | parsed typed actions, proposal edges, parse errors |
| `dpa_traces.jsonl` | per-belief DPA final statuses + audit trace |
| `metrics.json` | aggregate correctness metrics |
| `failure_breakdown.csv` | failure-mode tallies |
| `manifest.json` | run provenance (mode, model, prompt/template hashes, parser/schema versions) |

## Determinism guarantees

- `authorize(...)` and DPA are API-free and deterministic: identical inputs →
  identical authorized snapshot and trace.
- Subagent submissions are ordered deterministically before commit.
- Mock / smoke modes are fully offline and stable across machines.
- The `manifest.json` records prompt-template and dataset hashes so a live run
  can be tied to the exact inputs that produced it.

## What is never committed

Per `AGENTS.md`: `outputs/`, `reference/`, caches, local environments, generated
artifacts, benchmark downloads, checkpoints, adapter weights, and API keys.

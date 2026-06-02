# Artifact Note

A neutral artifact summary of what runs, how to reproduce it, and what is and
is not claimed.

## One-command offline reproduction

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests experiments
python3 -m pytest                       # full offline suite, no API key

python3 scripts/evaluate.py stage-ab --mock --max-cases 8 --constrained \
    --output-dir outputs/runs/smoke_stageab
python3 scripts/evaluate.py stage-c --smoke --max-cases 4 \
    --output-dir outputs/runs/smoke_stagec
```

No credentials are needed for the offline path. Live runs add `--live` with a
provider config (see [`api_providers.md`](api_providers.md)).

## What the artifact demonstrates

* **Deterministic authorization kernel.** `authorize(...)` is the sole public
  entrypoint; RevisionGate and DPA are internal. The docstring maps code to the
  paper formalism `G_t = (E_t, B_t, C_t, D_t, R_t)`, typed actions `A_t`, gate
  `Γ`, append-only update, and DPA status `σ_t(b)`.
* **Actions ≠ final status.** The parser rejects a final DPA status used as a
  typed action, rejects invalid target ids, enforces `BLOCKS`/`RELEASES` →
  conditions only, requires a grounded replacement for `SUPERSEDES`, and forbids
  mixing `NO_REVISION` with other actions (all fail-closed).
* **Fair Stage A vs Stage B vs Stage C** over identical fixed-candidate
  contexts; the commit/DPA path is shared and unchanged across stages.
* **Reproducibility.** Each run emits a manifest (git commit, provider, model,
  temperature, seed, prompt-template hash, parser version, dataset split) plus
  raw/parsed/DPA-trace JSONL.

## Mapping claims to evidence

| Claim | Evidence in repo |
| --- | --- |
| Deterministic precedence | `tests/gate_unit/test_defeat_path_authorization.py`, smoke item 9 |
| Typed-action validation fails closed | `tests/test_retrace_smoke.py` items 1–7 |
| Commit emits auditable gate/DPA trace | smoke item 8; `multiagent/commit.py` |
| Stage B canonicalization ≠ strict metric | smoke item 10; `metrics.py` |
| Provider-agnostic LLM layer | `tests/test_provider_modes.py` |

## What is **not** claimed

* No live Stage A-vs-B divergence table is reported here — offline mock/replay is
  a deterministic oracle replay (both stages reach 1.0 on the smoke cases), and a
  genuine divergence analysis requires a live model run that was not executed.
  Results are not fabricated.
* Stage C live / API-ICL is not turnkey; the API-ICL proposer is fail-closed
  without human-approved exemplars (`AGENTS.md` training boundary).
* Latent memory, RL consolidation, learned forgetting, and delayed-future-utility
  learning are future-scope work and intentionally absent.

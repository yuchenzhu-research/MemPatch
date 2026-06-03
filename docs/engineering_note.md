# Engineering Note

**Framing: a research prototype with production-minded engineering design — not
a production-ready system.** This note describes the engineering surface; the
research claims live in [`research_positioning.md`](research_positioning.md).

## What the engineering story is

ReTrace decides which subagent memory revisions are allowed to change a shared
usable memory basis, and emits an auditable trace per decision. The reliability
argument for multi-agent shared memory is: **the model only proposes; admission
and final status are decided by a deterministic, API-free kernel.** The same
inputs always produce the same committed snapshot.

## Clean provider abstraction

* One provider layer (`src/retracemem/providers/`) covers OpenAI-compatible,
  Anthropic Messages, and Ollama via a `mode` field — the runner is agnostic to
  the backend. See [`api_providers.md`](api_providers.md).
* Provider selection is config-driven: `--provider <name>` (registry) or
  `--provider-config <file>`. `--model` stays authoritative.
* **No hard-coded secrets.** Only `api_key_env` (the variable *name*) is stored;
  keys resolve from the environment / a git-ignored `.env`. Missing keys fail
  closed with a clear error. One transport helper redacts secrets from logs.

## Reproducible CLI

```bash
python3 scripts/evaluate.py stage-a --mock --max-cases 8 --constrained \
    --output-dir outputs/runs/smoke_stageab
```

Each run writes `manifest.json` (git commit, provider, model, temperature, seed,
prompt-template hash, parser version, dataset provenance), `metrics.json`,
`failure_breakdown.csv`, and raw/parsed/DPA-trace JSONL. Offline `--mock` /
`--smoke` modes need no credentials, so CI and local dev never touch a network.

## Modular structure

```text
src/retracemem/
  authorization.py        # authorize(...) — sole public kernel (deterministic)
  tms/                    # RevisionGate (admission) + DPA (status)
  multiagent/             # commit wrappers around authorize(...)
  proposers/              # typed-action proposers (Stage A / C)
  providers/              # provider-agnostic LLM layer (this note)
  evaluation/multiagent/  # shared runner, metrics, manifests
```

Core DPA logic is standard-library-first and API-free; provider/benchmark logic
is isolated in proposers / runners / adapters.

## Tests & verification

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests experiments
python3 -m pytest        # full offline suite
```

The suite includes provider-mode tests (payload shape + response parsing per
mode, key-resolution fail-closed) and consolidated parser / gate / DPA / commit
smoke tests, all offline.

## Honest limitations

* Live multi-provider paths are implemented but only the OpenAI-compatible path
  has been exercised against a real endpoint here; Anthropic/Ollama are covered
  by offline transport-mocked tests.
* Stage C runs offline (mock / decoded-generation replay) only; API-ICL is not
  wired to a turnkey live CLI. Not production-hardened (no rate-limit/back-pressure
  orchestration, no multi-tenant concerns).

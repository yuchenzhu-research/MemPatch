# ReTrace Project Governance

This is the canonical governance document for the repository. It defines the
umbrella project, the active research tracks, the role of ReTrace-Engine, the
inactive/backlog items, and the reference-management policy. When other docs
disagree about project structure, this document is authoritative.

## Umbrella project: ReTrace

**ReTrace** is the umbrella project for reliable shared-memory revision in
multi-agent/agentic workflows. It is governed as **two active research tracks**
— not three or four. ReTrace-Engine is a submodule of one track, not a track of
its own.

## Active track 1 — ReTrace-Bench (benchmark track)

Benchmark paper / benchmark track. Evaluation-only and independent of any
training method: any memory-enabled agent (LLM-only, RAG, CRUD store,
Mem0-style, or a trained policy) can be scored on it.

Owns: benchmark data, scenario schema, scoring/metrics, baselines, the four
v1.0 evaluation splits (`main`/`hard`/`realistic`/`calibration`), leakage
checks, and benchmark-paper framing.

Canonical locations:

- `benchmark/retrace_bench/`
- `data/retrace_bench/` (v1.0 splits `main_3000_en`, `hard_300_en`,
  `realistic_100_en`, `calibration_80_en`)
- `docs/retrace_bench/`
- benchmark-related scripts (`scripts/generate_retrace_main_3000.py`,
  `scripts/generate_retrace_hard_300.py`, `scripts/run_retrace_bench_*.py`,
  `scripts/validate_retrace_bench_dataset.py`) and `tests/retrace_bench/`.

Learning-only training/validation datasets live under
`data/retrace_learn/v1_0/` and are owned by the ReTrace-Learn track,
not the benchmark. Pre-v1 supervision scaffolding was removed from the active tree due to leakage issues and is recoverable only from Git history.

## Active track 2 — ReTrace-Learn (method track)

Method paper / trainable method track. ReTrace-Learn v1 has **three
paper-facing stages** (only the first two are learned):

```text
Graph Builder  ->  Proposal Policy  ->  DPA-guided RSFT / DPO
  (learned)          (learned)            (training protocol)
```

1. **Graph Builder** — raw dialogue / memory snapshot -> candidate memory graph.
2. **Proposal Policy** — candidate graph + new evidence -> typed revision proposal.
3. **DPA-guided RSFT / DPO** — DPA verifies/filters/ranks proposals and creates
   the RSFT/DPO training signals for the Proposal Policy. This is a training
   *protocol*, not a trainable module; DPA itself does not learn.

The deterministic commit path — **ReTrace-Engine** (Parser + RevisionGate + DPA
+ Audit Trace), reached via the single public entrypoint `authorize(...)` — is an
*implementation detail* of stages 2–3, not a separate paper-level module. The
final memory commit is deterministic and API-free.

Canonical locations:

- `src/retrace_learn/` — learned stages (data, runtime, training).
- `src/retracemem/` — the deterministic ReTrace-Engine:
  `authorize(...)`, RevisionGate, Defeat-Path Authorization (DPA), schemas.
- method training/evaluation scripts (`scripts/evaluate.py`,
  `scripts/export_stagec_data.py`, etc.) and method docs
  (`docs/architecture.md`, `docs/retrace_learn_pipeline.md`,
  `docs/experiment_protocol.md`).

### ReTrace-Engine is a submodule, not a track

- Use "ReTrace-Engine" **only** as the implementation name for the deterministic
  commit path inside ReTrace-Learn.
- ReTrace-Engine is **not** a standalone paper, **not** a standalone top-level
  research track, and **not** a third big module.
- `authorize(...)` is the sole public authorization entrypoint; neither DPA nor
  RevisionGate is invoked directly by external callers.

## Inactive / backlog (not active scope)

The following are explicitly **out of active scope** for the current two-track
governance. Do not add code or active docs for them; mention only as backlog.

- **ReTrace-SkillOpt / frozen-agent skill optimization** — not an active
  paper/track.
- **`memory_policy.md` optimization** — not active work.
- **Microsoft SkillOpt integration** — not active work.
- **Future latent-memory line** — latent/hidden memory state, learned
  forgetting, RL consolidation, delayed-future-utility learning, and
  biological-memory mechanisms are reserved for a future paper and must not
  appear in this codebase as active scope.

## Reference-management policy

- **Pointer-style only.** External references are stored as YAML registry
  entries (`references/**/registry.yaml`) plus short Markdown notes
  (`references/**/notes/*.md`) with URLs / paper links / repo links.
- **No vendoring.** Do not vendor or commit external repositories, cloned source
  trees, or PDFs.
- **Ignored local clones.** If a local clone is needed for diagnostics, it must
  live under an ignored directory (`.external_repos/` or `.reference_cache/`)
  and must never be committed. Do not edit or commit those directories except to
  remove accidentally tracked content.
- If an external clone or large source is ever found tracked, remove it from git
  and replace it with one registry entry plus one short note file.

## Do-not-commit (preserved from `AGENTS.md`)

Never commit external clones, `outputs/`, caches, local environments, generated
artifacts, benchmark downloads, model weights/checkpoints, or API keys.

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

Owns: benchmark data, scenario schema, scoring/metrics, baselines, the held-out
test split, leakage checks, and benchmark-paper framing.

Canonical locations:

- `benchmark/retrace_bench/`
- `data/retrace_bench/`
- `data/retrace_supervision/`
- `docs/retrace_bench/`
- benchmark-related scripts (`scripts/generate_retrace_data_splits.py`,
  `scripts/check_retrace_split_leakage.py`, `scripts/run_retrace_bench_*.py`,
  `scripts/validate_retrace_bench_dataset.py`) and `tests/retrace_bench/`.

## Active track 2 — ReTrace-Learn (method track)

Method paper / trainable method track. Owns the three-part method pipeline:

```text
Graph Extractor  ->  Typed Revision Proposer  ->  Authorization Court
   (learned)              (learned)                 (deterministic)
```

The **Authorization Court** is implemented by **ReTrace-Engine** via the single
public entrypoint `authorize(...)`. The final memory commit is deterministic and
API-free.

Canonical locations:

- `src/retrace_learn/` — learned modules (data, runtime, training).
- `src/retracemem/` — the deterministic Authorization Court (ReTrace-Engine):
  `authorize(...)`, RevisionGate, Defeat-Path Authorization (DPA), schemas.
- method training/evaluation scripts (`scripts/evaluate.py`,
  `scripts/export_stagec_data.py`, etc.) and method docs
  (`docs/architecture.md`, `docs/retrace_learn_pipeline.md`,
  `docs/experiment_protocol.md`).

### ReTrace-Engine is a submodule, not a track

- Use "ReTrace-Engine" **only** as the implementation name for the deterministic
  Authorization Court inside ReTrace-Learn.
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

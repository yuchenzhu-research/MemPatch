# Model Context Index

This document coordinates Codex, Gemini, Opus, and any future coding model. Its
job is to reduce context drift when models have limited memory or different code
styles.

## Why The Source Materials Are In This Repository

Two planning documents are now tracked under `docs/source_materials/`:

- `iclr_2027_paper_1_final_blueprint_re_trace.md`
- `re_trace_companion_codebase_integration_and_model_handoff.md`

They are complementary:

- The blueprint is the scientific specification: paper identity, method scope,
  benchmarks, contribution boundaries, math, figures/tables, and experiment
  strategy.
- The companion is the implementation and handoff contract: how to inspect
  reference repositories, how to wrap external code, how to avoid model drift,
  and how to keep results reproducible.

Both files should be read by every major model session. They are intentionally
kept as raw source materials and should not be edited during implementation.

## Authority Order

Use this order when files disagree:

1. Current repository code and tests.
2. `docs/coding_contract.md` for implementation style and local constraints.
3. `docs/project_logic.md` for current research interpretation.
4. `docs/implementation_status.md` for completed vs total first-version scope.
5. `docs/source_materials/iclr_2027_paper_1_final_blueprint_re_trace.md` for
   scientific claims, method boundaries, benchmarks, and paper framing.
6. `docs/source_materials/re_trace_companion_codebase_integration_and_model_handoff.md`
   for upstream intake, reproducibility discipline, and handoff intent.
7. External repositories under `reference/`.

Important adaptation: the companion suggests an `external/` layout in places.
This repository currently uses ignored local clones under `reference/`. Keep
using `reference/` unless the project explicitly migrates.

## Current Codebase State

The first runnable research loop exists:

```text
data/boundary_audit/minimal.jsonl
→ retracemem.verifier.HeuristicRelationVerifier
→ retracemem.pipeline.ReTracePipeline
→ retracemem.tms.RevisionGate
→ retracemem.tms.AuthorizationEngine
→ retracemem.generation.BasisBuilder
→ retracemem.schemas.EvaluationRecord
→ scripts/run_boundary_audit.py
```

Smoke runners also exist:

- `scripts/run_stale.py`
- `scripts/run_memora.py`

The project has not yet implemented full official benchmark scoring, prompt
verification, learned verification, or heavy baseline wrappers. Those belong to
later scoped work.

## Paper Method In Code Terms

Use this mapping when writing code:

| Paper concept | Code location |
|---|---|
| Immutable episodic evidence | `retracemem/memory/episode_ledger.py` |
| Open-text belief nodes | `retracemem/memory/belief_store.py` |
| Relation verifier | `retracemem/verifier/` |
| Conservative revision authorization | `retracemem/tms/gate.py` |
| Current belief decision | `retracemem/tms/authorization.py` |
| Query-time authorized basis | `retracemem/generation/basis_builder.py` |
| End-to-end local method loop | `retracemem/pipeline.py` |
| Unified output | `retracemem/schemas.py`, `retracemem/evaluation/` |
| Diagnostic cases | `data/boundary_audit/minimal.jsonl` |
| Benchmark adapters | `retracemem/adapters/` |
| Benchmark smoke runners | `scripts/run_stale.py`, `scripts/run_memora.py` |

## Reference Repository Usage

Reference repositories are local, ignored clones under `reference/`.

Priority references:

- `reference/STALE`
  - benchmark and CUPMem baseline source;
  - do not inherit CUPMem's fixed slot ontology as ReTrace core.
- `reference/Memora`
  - evolving-memory benchmark and FAMA source;
  - use as benchmark/evaluator, not method template.
- `reference/mem0`
  - useful direct wrapper candidate later;
  - add explicit provenance metadata if wrapped.
- `reference/nemori`
  - episode/semantic provenance and optional baseline;
  - do not convert ReTrace into distillation.
- `reference/graphiti`
  - temporal provenance and fact validity concepts;
  - do not make graph infrastructure dominate Paper 1.
- `reference/TriMem`
  - source dialogue IDs and atomic fact design.

Do not vendor external repository code into ReTrace. Prefer adapters, wrappers,
and notes that record upstream commit SHA and role.

## Model-Specific Work Routing

### Gemini-3.5 Flash

Best for:

- small deterministic edits;
- JSONL fixture additions;
- CLI argument polish;
- doc updates;
- test additions around existing functions;
- compile-error fixes.

Avoid assigning:

- architecture redesign;
- benchmark-scoring integration;
- relation-verifier semantics changes without tests;
- reference repository interpretation.

### Opus 4.7

Best for:

- multi-file integration;
- prompt verifier implementation;
- official evaluator wrappers;
- baseline wrapper design;
- consistency review across docs, scripts, and tests.

Guardrails:

- do not add heavy dependencies in core modules;
- do not replace current ReTrace method with a reference framework;
- keep changes scoped and commit in English.

### Codex

Best for:

- repository surgery;
- commit/push workflow;
- adapter and runner integration;
- test and smoke-command execution;
- enforcing codebase-wide consistency.

## How To Start A New Model Session

Use this prompt skeleton:

```text
You are working in ReTrace. First read AGENTS.md and docs/model_context_index.md.
Then read the two source documents under docs/source_materials/.
Follow docs/coding_contract.md and docs/implementation_status.md.
Implement only the scoped task below.
Do not reinterpret the research direction.
Do not edit reference/.
Keep the code standard-library-first and no-API in core logic.
Run compileall before reporting.

Task: ...
```

## What Counts As Done For A Scoped Task

A scoped task is done only when:

- code is implemented in the correct package boundary;
- tests or smoke checks cover the behavior;
- unified JSONL output is preserved if the task produces method outputs;
- docs are updated only when behavior or status changes;
- `compileall` passes;
- git status is clean after commit.


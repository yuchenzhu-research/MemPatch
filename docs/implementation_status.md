# Implementation Status

Last updated: 2026-05-27

This document tracks the first research-code version. It intentionally has only
two sections:

- Total plan: the agreed implementation surface for the first complete version.
- Completed: what is already implemented in this repository.

Anything in the total plan that is not listed as completed is remaining work by
definition.

## Total Plan

### Research Alignment

- Lock ReTrace as evidence-preserving reversible belief revision.
- Keep external reference repositories isolated under `reference/`.
- Keep Paper 1 focused on STALE, Memora, and a small BoundaryAudit diagnostic
  set.
- Avoid latent memory, RL/GRPO, destructive consolidation, and fixed-slot
  ontology drift.

### Core Data Contracts

- Stable dataclass schemas for evidence, beliefs, relation predictions,
  authorization decisions, and evaluation records.
- Unified JSONL output for all methods.
- Cost/call tracking helpers.

### Memory Core

- Append-only `EpisodeLedger`.
- Open-text `BeliefStore`.
- Conservative `RevisionGate`.
- `AuthorizationEngine` for deciding whether a belief can govern current
  answers.
- `BasisBuilder` for query-time authorized basis construction.

### Verifier Layer

- Deterministic `HeuristicRelationVerifier` for local smoke runs.
- Relation labels:
  - `SUPPORT`
  - `SUPERSEDE`
  - `BLOCK`
  - `CONDITION`
  - `NONE`
  - `UNCERTAIN`
- Fail-closed behavior for ambiguous or empty inputs.

### ReTrace Pipeline

- `ReTracePipeline` that combines:
  - evidence ledger;
  - belief store;
  - verifier;
  - revision gate;
  - authorized basis;
  - deterministic evaluation record output.

### BoundaryAudit Diagnostic Loop

- 20 local JSONL diagnostic cases.
- Balanced case buckets:
  - `SUPERSEDE`
  - `BLOCK`
  - `CONDITION`
  - `NONE`
  - `UNCERTAIN`
- Runner for:
  - `retrieval_baseline`
  - `retrace_heuristic`
- Summary metrics:
  - `cases_total`
  - `relation_correct`
  - `authorization_correct`
  - `protected_beliefs_preserved`
  - `unsupported_revision_count`

### Benchmark Smoke Runners

- STALE smoke runner:
  - discover `*_MAIN.json`;
  - load a limited number of samples;
  - run retrieval baseline;
  - write unified JSONL.
- Memora smoke runner:
  - discover persona roots;
  - load chronological sessions and questions;
  - run retrieval baseline;
  - write unified JSONL.
- Empty/missing benchmark data should exit cleanly with an empty JSONL file.

### Tests And Verification

- No-dependency compile check:

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile python3 -m compileall -q retracemem tests scripts
```

- Unit tests for:
  - adapters;
  - memory core;
  - TMS authorization;
  - retrieval baseline;
  - evaluation helpers;
  - heuristic verifier;
  - ReTrace pipeline;
  - runners.

## Completed

### Research Alignment

- Added `docs/project_logic.md`.
- Added `docs/coding_contract.md`.
- Added `docs/agent_handoff.md`.
- Added `docs/today_execution_plan.md`.
- Added `docs/reference_integration_map.md`.
- Added `docs/code_direction.md`.

### Core Data Contracts

- Implemented `retracemem/schemas.py`.
- Implemented `retracemem/evaluation/jsonl.py`.
- Implemented `retracemem/evaluation/cost_tracker.py`.
- Implemented `retracemem/evaluation/records.py`.

### Memory Core

- Implemented `retracemem/memory/episode_ledger.py`.
- Implemented `retracemem/memory/belief_store.py`.
- Implemented `retracemem/tms/gate.py`.
- Implemented `retracemem/tms/authorization.py`.
- Implemented `retracemem/generation/basis_builder.py`.

### Verifier Layer

- Implemented `retracemem/verifier/heuristic_verifier.py`.
- Exported `HeuristicRelationVerifier` from `retracemem/verifier/__init__.py`.
- Added `tests/test_heuristic_verifier.py`.

### ReTrace Pipeline

- Implemented `retracemem/pipeline.py`.
- Added `tests/test_pipeline.py`.

### BoundaryAudit Diagnostic Loop

- Added `data/boundary_audit/minimal.jsonl`.
- Implemented `scripts/run_boundary_audit.py`.
- Added runner coverage in `tests/test_runners.py`.
- Verified:

```text
python3 scripts/run_boundary_audit.py --method retrace_heuristic --output /tmp/retrace_heuristic.jsonl
cases_total: 20
relation_correct: 20
authorization_correct: 20
protected_beliefs_preserved: 20
unsupported_revision_count: 0
```

- Verified retrieval baseline comparison:

```text
python3 scripts/run_boundary_audit.py --method retrieval_baseline --output /tmp/retrieval_baseline.jsonl
cases_total: 20
relation_correct: 0
authorization_correct: 8
protected_beliefs_preserved: 20
unsupported_revision_count: 0
```

### Benchmark Smoke Runners

- Implemented `scripts/run_stale.py`.
- Implemented `scripts/run_memora.py`.
- Verified STALE empty-data behavior against the current reference clone:

```text
python3 scripts/run_stale.py --limit 3 --method retrieval_baseline --output /tmp/stale_retrieval.jsonl
No STALE MAIN files found under reference/STALE; nothing to run.
records_written: 0
```

- Verified Memora smoke on the current reference clone:

```text
python3 scripts/run_memora.py --limit 3 --method retrieval_baseline --output /tmp/memora_retrieval.jsonl
sessions_loaded: 619
questions_loaded: 3
records_written: 3
```

### Tests And Verification

- Added or retained tests:
  - `tests/test_adapters.py`
  - `tests/test_evaluation_helpers.py`
  - `tests/test_memory_core.py`
  - `tests/test_retrieval_backend.py`
  - `tests/test_tms_authorization.py`
  - `tests/test_heuristic_verifier.py`
  - `tests/test_pipeline.py`
  - `tests/test_runners.py`

- Verified no-dependency compile check:

```text
env PYTHONPYCACHEPREFIX=/Users/yuchenzhu/Desktop/ReTrace/.pycache_compile python3 -m compileall -q retracemem tests scripts
passed
```

- `pytest` is not installed in the current environment, so pytest execution was
  not used for final verification.


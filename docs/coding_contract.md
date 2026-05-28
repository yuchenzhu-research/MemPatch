# Coding Contract

This document defines package boundaries, editing rules, testing rules, and
provenance rules for ReTrace.

## General Rules

- Use Python 3.10+.
- Prefer the standard library.
- Do not add heavy dependencies unless an approved stage explicitly requires
  them.
- Do not introduce API calls in core logic.
- Do not modify code under `reference/`.
- Do not commit `reference/`, `outputs/`, caches, virtual environments, API
  keys, benchmark downloads, or generated run artifacts.
- Keep functions small and deterministic.
- Preserve existing schema field names unless an approved migration updates all
  users and tests.
- Do not add notebook-style exploratory code to package modules.

## Canonical Runtime Boundary

New runtime method work must use the typed graph and DPA contracts:

- `EvidenceNode`
- `BeliefNode`
- `ConditionNode`
- `DependencyEdge(REQUIRES)`
- `EvidenceEdge(BLOCKS / RELEASES / SUPERSEDES / REAFFIRMS / UNCERTAIN)`
- `DefeatPath`
- `AuthorizationTrace`

Legacy flat `RelationPrediction` semantics, including `SUPPORT`, `CONDITION`,
and `REQUIRED_BY`, may remain only as transitional compatibility or retired
historical references. They must not govern new method code, prompts, runners,
or paper-facing documentation.

Development-only heuristic/manual fixtures may be used for tests and smoke
runs. They are forbidden as paper main-result methods.

## Package Boundaries

### `src/retracemem/schemas.py`

Owns stable dataclass contracts. Do not add benchmark-specific fields directly
to canonical dataclasses; put benchmark-specific data in `metadata`.

### `src/retracemem/memory`

Owns local memory storage:

- append-only evidence ledger;
- open-text belief and condition store;
- typed dependency and evidence edge collections.

### `src/retracemem/tms`

Owns deterministic authorization:

- `RevisionGate` structurally admits typed edges;
- `DefeatPathAuthorizationAlgorithm` computes final belief authorization;
- no semantic-model calls;
- no answer generation.

### `src/retracemem/verifier`

Owns local typed proposal interfaces:

- `RequirementInducer`;
- `EvidenceEdgeVerifier`;
- `PromptTypedBeliefExtractor`;
- `PromptRequirementInducer`;
- `PromptEvidenceEdgeVerifier`;
- development-only manual/heuristic fixtures.

Verifiers propose local objects. They do not mutate memory and do not decide
final authorization.

### `src/retracemem/methods`

Owns controlled Stage A/B method paths:

- `SharedCandidateView`;
- `ControlledReTraceLLM`;
- `DirectJudgeLLM`;
- controlled method result records.

`DirectJudgeLLM` is a sibling direct-adjudication baseline, not an
`EvidenceEdgeVerifier` and not a DPA path.

### `src/retracemem/providers`, `src/retracemem/cache`, accounting modules

Own the only allowed provider/cache/accounting boundary for future live model
calls. Random modules must not import SDK clients or call live models directly.

The current documentation reset must not connect a provider or call a live API.

### `src/retracemem/retrieval`, `src/retracemem/backends`, `src/retracemem/pipeline`

Own retrieval, backend, and pipeline integration. Keep benchmark-specific logic
out of core DPA and verifier contracts.

### `src/retracemem/adapters`

Own benchmark loading and normalization. Adapters must degrade cleanly on
missing local reference data and must not mutate `reference/`.

### `scripts`

Runner scripts should:

- parse CLI args with `argparse`;
- avoid API keys for offline smoke or replay mode;
- write outputs under ignored output/artifact locations;
- print compact summaries;
- never mutate files under `reference/`.

## External Repositories

`reference/` is the canonical ignored local-upstream clone location. Preserve
this convention unless a future task explicitly authorizes a migration.

Do not copy upstream implementation code into ReTrace. Prefer wrappers,
adapters, provenance records, and patch files when unavoidable.

## Prompt and Provenance Rules

- Version prompt templates; never silently overwrite a prompt used for a run.
- Record prompt hash, parser/schema version, model id, provider, model
  revision/API version when available, cache behavior, calls, tokens, latency,
  and errors.
- Preserve `model_call_trace_id` for paper-relevant semantic calls.
- Record gate rejections and parser failures instead of hiding them.

## Test and Environment Rules

Compile:

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests scripts
```

Full offline tests:

```bash
.venv/bin/python -m pytest
```

If a test needs temporary files, use `tempfile` or pytest `tmp_path`. Do not
write temporary cache artifacts directly to tracked directories.

## Commit Rules

Use short English commit messages with production-level scope:

- `Add ...`
- `Implement ...`
- `Document ...`
- `Wire ...`
- `Fix ...`

Do not bundle unrelated method, runner, and documentation changes into one
commit unless the change is purely mechanical.

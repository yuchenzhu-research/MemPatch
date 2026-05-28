# Coding Contract

This document defines how code should be written in this repository. Follow it
even when using a different model or coding assistant.

## General Rules

- Use Python 3.10+.
- Prefer the standard library.
- Do not add heavy dependencies unless a plan explicitly names them.
- Do not introduce API calls in core logic.
- Do not modify code under `reference/`.
- Do not commit `reference/`, outputs, caches, virtual environments, API keys,
  or benchmark downloads.
- Keep functions small and deterministic.
- Use dataclasses for shared records.
- Preserve existing schema field names unless a migration note is added.
- Do not add notebook-style exploratory code to package modules.

## Dependency Rules

Allowed now:

- Python standard library.
- Existing local package modules.

Not allowed in first-version core:

- `torch`
- `transformers`
- `langchain`
- `llama-index`
- `networkx`
- `pandas`
- `numpy`
- benchmark-specific packages outside wrappers

If a later baseline wrapper needs a dependency, isolate it in a wrapper module
and keep the no-dependency tests runnable.

## Package Boundaries

### `retracemem.schemas`

Owns stable data contracts:

- `EvidenceNode`
- `BeliefNode`
- `ConditionNode`
- `DependencyEdge`
- `EvidenceEdge`
- `DefeatPath`
- `AuthorizationTrace`
- `RequirementProposal` (in verifier/contracts.py)

Legacy types (`EpisodicEvidence`, `Belief`, `RelationPrediction`, `AuthorizationDecision`, `EvaluationRecord`) are kept only for transitional compatibility; new runtime code must not import or return them. Do not add benchmark-specific fields directly to these dataclasses. Put benchmark-specific data in `metadata`.

### `retracemem.memory`

Owns local memory storage:

- append-only evidence ledger;
- open-text belief store;
- no fixed domain slots.

### `retracemem.verifier`

Owns relation prediction contracts:

- `RequirementInducer` and `EvidenceEdgeVerifier` protocols.
- `HeuristicRequirementInducer` and `HeuristicEvidenceEdgeVerifier` (development-only deterministic fixtures).
- `PromptEvidenceEdgeVerifier` (LLM-based verifier).

No verifier should directly mutate memory. Deep learning or heavy machine learning packages (e.g., PyTorch, Transformers) must not be added as core dependencies; they are deferred and may only be introduced as optional extensions for a future `ReTrace-Local` edge verifier.

### `retracemem.tms`

Owns revision authorization:

- deterministic revision gate (`RevisionGate`);
- decide current authorization via Defeat-Path Authorization (`DPA`);
- keep blocked beliefs auditable via `DefeatPath` and `AuthorizationTrace`.

The TMS layer decides whether a belief may govern current answers. It does not generate prose answers.

### `retracemem.generation`

Owns query-time basis construction and answer wrappers.

The first version may return deterministic answer shells. Real LLM answerers must be wrapper-only and optional.

### `retracemem.adapters`

Owns benchmark loading and normalization.

Adapters must degrade cleanly:

- missing reference roots return empty discovery results;
- invalid JSON returns empty records;
- missing optional fields become empty strings/lists/metadata.

### `retracemem.backends`

Owns method wrappers behind one interface:

```python
reset_user(user_id)
ingest_session(user_id, session, metadata=None)
search(user_id, query, limit=10, metadata=None)
answer(user_id, query, retrieved, metadata=None)
```

### `retracemem.evaluation`

Owns unified JSONL records, cost tracking, and metric helpers.

All methods should emit `EvaluationRecord` or a JSON-compatible equivalent.

## Runner Rules

Runner scripts under `scripts/` should:

- parse CLI args with `argparse`;
- not require API keys for smoke mode;
- write JSONL under `outputs/`;
- print a compact summary;
- never mutate files under `reference/`.

## Test and Environment Rules

- Python Version: Standard package requires Python >= 3.10. Current development environment uses a project-local virtual environment based on Python 3.10.20.
- Setup: Create virtual environment and install in editable mode:
  ```bash
  ~/miniconda3/envs/paper/bin/python -m venv .venv
  .venv/bin/python -m pip install -e ".[dev]"
  ```
- Preferred command to run tests:
  ```bash
  .venv/bin/python -m pytest
  ```
- Compilation check:
  ```bash
  env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests scripts
  ```
- If a test needs temporary files, use `tempfile` or pytest `tmp_path`. Do not write temporary cache artifacts directly to `tests/` or track them in Git.

## Style Rules

- Use clear engineering names over paper-jargon names.
- Keep comments sparse and useful.
- Use explicit error messages for rejected invalid states.
- Fail closed for unclear verifier output.
- Prefer returning empty lists over raising for missing benchmark data.
- Keep all timestamps as strings until a specific temporal parser is needed.


# ReTrace Agent Instructions

This is the first file every coding model must read before editing this repository.

## Current State

Dynamic branch, HEAD, smoke, and validation status live only in `README.md`. This file contains stable agent instructions, not volatile run status.

## Canonical Reading Order

Read only these active authority documents before method work:

1. `AGENTS.md`
2. `README.md`

Legacy planning documents and old raw source-material files are no longer active authority. Git history preserves them.

## One-Sentence Alignment

ReTrace preserves immutable evidence and changes a belief's eligibility for current answers only through verified, temporally valid typed defeat paths computed by deterministic DPA.

## Method Boundary / Identity

Paper 1 studies evidence-preserving reversible authorization for evolving agent memory:

```text
immutable evidence ledger
+ typed belief/condition/evidence-edge graph
+ deterministic Defeat-Path Authorization Algorithm
```

ReTrace restricts semantic-model judgment to local typed proposals and delegates final belief authorization to deterministic DPA. It is a pluggable authorization kernel, not a full memory system.

## One-Function Public API Boundary

The sole public entrypoint for executing authorization is `authorize(...)`:

```python
def authorize(
    view: SharedCandidateView,
    proposal_batches: tuple[EvidenceProposalBatch, ...],
    *,
    audit_metadata: dict[str, Any] | None = None,
) -> AuthorizationResult:
    ...
```

* Neither DPA nor RevisionGate should be invoked directly by external callers.
* All updates/admissions and deterministic routing happen entirely inside `authorize`.

## Canonical Runtime Vocabulary and Typed Edges

Only this typed scheme is canonical for method documentation and runtime work:

- `DependencyEdge(REQUIRES)`: belief -> condition.
- `EvidenceEdge(BLOCKS)`: evidence -> condition.
- `EvidenceEdge(RELEASES)`: evidence -> condition.
- `EvidenceEdge(SUPERSEDES)`: evidence -> prior belief, with grounded `replacement_belief_id`.
- `EvidenceEdge(REAFFIRMS)`: evidence -> belief.
- `EvidenceEdge(UNCERTAIN)`: evidence -> belief.

## Pluggable DPA Precedence

For any candidate belief `b`:
```text
A_t(b) = DPA(b, S_t) ∈ {AUTHORIZED, BLOCKED, SUPERSEDED, UNRESOLVED}
```
with canonical precedence:
```text
SUPERSEDES > PREREQUISITE_BLOCK > UNRESOLVED_UNCERTAIN > AUTHORIZED
```
Temporal tie-breaking must remain deterministic using canonical ordering data.

## Experiment Isolation Rules (STALE)

To ensure absolute clean methodology and avoid test-set leakage:
- STALE adapters, runners, and logic must reside strictly inside `experiments/` (e.g. `experiments/stale_adapter.py`).
- Methods (write-time, probe-time) must consume isolated interfaces (`StaleWriteHistory`, `StaleProbeTask`) that strip out gold fields.
- `StaleGoldRecord` containing `M_old`, `M_new`, `explanation`, conflict type, and `relevant_session_index` must be restricted to evaluation and scoring boundaries.
- All three probing queries must bind to the same frozen memory snapshot.

## Do Not Drift (No-Go List)

Do not turn this codebase into:
- generic RAG;
- a Mem0 clone;
- a Graphiti clone;
- CUPMem fixed-slot state tracking;
- RL memory action learning;
- latent memory, memory-token, or learned consolidation work;
- an unconstrained LLM judge that directly rewrites memory.

## Test and Clean-Worktree Rules

- Standard library first.
- Keep core DPA logic API-free and deterministic.
- Keep benchmark-specific logic in adapters or runners.
- Do not edit `reference/`.
- Do not commit `reference/`, `outputs/`, caches, local environments, generated artifacts, benchmark downloads, or API keys.
- Preserve the canonical dataclass contracts in `src/retracemem/schemas.py`.
- Add or update tests for every new behavior.

## Verification

Compile:
```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests experiments
```

Full offline tests:
```bash
.venv/bin/python -m pytest
```

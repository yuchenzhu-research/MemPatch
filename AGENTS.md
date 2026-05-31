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

Paper 1 is centered on multi-agent/subagent shared-memory revision authorization.

Multiple subagents may submit evidence-bearing memory updates to a shared long-term memory. ReTrace controls which revisions are allowed to affect the shared usable memory basis.

Stage naming and configuration hierarchy:
- Stage A = `ReTrace-API-ZeroShot` / `ReTrace-Prompt` (API model proposes typed revision actions over a fixed candidate view, then routes through ReTrace-Core. Do NOT embed a dialogue extractor in Stage A).
- Stage B = `DirectJudge-API` (API baseline model directly predicts final belief usability status, completely bypassing ReTrace-Core).
- Stage C-Fixed = `ReTrace-AdaptiveProposer-Fixed` (learned proposer replaces prompt proposer, evaluated over fixed candidate views to preserve controlled comparison).
- Stage C-Raw / ReTrace-Learn-Full = `ReTrace-Learn-Full` (main research direction: consumes raw dialogues/submissions -> extracts graph nodes/dependencies via a learned extractor -> proposes typed actions -> commits through deterministic ReTrace-Core).

Public API Boundaries:
- `authorize(...)` is the public deterministic authorization kernel. Neither DPA nor RevisionGate should be invoked directly by external callers. All updates/admissions and deterministic routing happen entirely inside `authorize`.
- `commit_subagent_submission(...)` and `commit_submission_sequence(...)` are multi-agent integration wrappers around `authorize(...)`.

STALE/CUPMem is an external validation/baseline pathway, not the definition of the paper.

Latent memory, RL consolidation, and delayed-utility learning belong to Paper 2.

## Paper 1 Stage C Training Boundary

Paper 1 includes Stage C: learning an explicit typed revision proposal policy
for multi-agent/subagent shared-memory updates.

The Stage C policy consumes only method-visible inputs:
a prior shared-memory context or bounded candidate view, an evidence-bearing
subagent submission, candidate beliefs/replacements, conditions, and
pre-existing dependency anchors.

It proposes explicit revision actions from the canonical vocabulary:

- `SUPERSEDES`
- `BLOCKS`
- `RELEASES`
- `UNCERTAIN`
- `REAFFIRMS`
- `NO_REVISION`

Final memory commit remains deterministic and API-free:

```text
Stage C policy proposal
    -> RevisionGate
    -> deterministic DPA / authorize(...)
    -> SharedMemoryCommitResult
```

Paper 2, not Paper 1, owns latent-memory representations,
long-horizon delayed-future-utility consolidation, and RL over hidden memory states.
Paper 1 may later test short-horizon explicit-action refinement only if it
does not introduce latent memory or hidden-state consolidation.

A Stage C live policy may see conditions and pre-existing REQUIRES anchors
because they are method-visible candidate structure, but may never see
typed gold revision targets or evaluator final statuses.

Only human-approved reviewed examples may be used for live smoke or training export.

No development-candidate episode may be promoted for smoke or training until a human review decision has been recorded. All live Stage C actions must explicitly cite the visible new evidence that grounds the proposed revision.

Approval for prompt smoke or training must be derived from an explicit human decision record tied to an immutable review-pack manifest hash. Scripts must not automatically promote pending examples to approved.

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

- do not turn the repository into a generic orchestration framework;
- do not implement agent debate/voting;
- do not duplicate the `authorize(...)` kernel;
- do not change DPA semantics without a demonstrated deterministic bug;
- do not leak STALE gold fields into method inputs;
- do not use official scored examples for prompt tuning.
- do not turn this codebase into generic RAG, Mem0 clone, or Graphiti clone.

## Test and Clean-Worktree Rules

- Standard library first.
- Keep core DPA logic API-free and deterministic.
- Keep benchmark-specific logic in adapters or runners.
- Do not edit `reference/`.
- Do not commit `reference/`, `outputs/`, caches, local environments, generated artifacts, benchmark downloads, or API keys.
- Preserve the canonical dataclass contracts in `src/retracemem/schemas.py`.
- Add or update tests for every new behavior.

## Paper Experiment Hierarchy

E0 — Oracle/Replay Kernel Validation:
     hand-authored typed proposals; engineering/mechanism verification only.

E1 — Fixed-Candidate Revision Evaluation:
     same evidence/candidate context for all methods; methods must decide revisions;
     primary controlled method comparison.

E2 — Stage C Training and Model-Driven Proposal Evaluation:
     training and evaluating learning-based proposal policies.

E3 — Closed-Loop Multi-Agent Workflow:
     shared memory affects downstream agent actions and future submissions.

E4 — STALE/CUPMem External Validation:
     external stale-memory validation and compatibility analysis.

Do not let external STALE/CUPMem bridge code redefine the primary Paper 1
method identity or the main evaluation data model.

## Verification


Compile:
```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests experiments
```

Full offline tests:
```bash
.venv/bin/python -m pytest
```

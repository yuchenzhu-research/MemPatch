# ReTrace Agent Instructions

This is the first file every coding model must read before editing this repository.

## Current State

Dynamic branch, HEAD, smoke, and validation status live only in `README.md`. This file contains stable agent instructions, not volatile run status.

## Canonical Reading Order

Read only these active authority documents before method work:

1. `AGENTS.md`
2. `README.md`
3. `docs/project_governance.md` (canonical two-track project structure)

Legacy planning documents and old raw source-material files are no longer active authority. Git history preserves them.

## Active Research Tracks

ReTrace is an umbrella project governed as **two active research tracks** (see `docs/project_governance.md`):

1. **ReTrace-Bench** — benchmark track. Evaluation-only; owns benchmark data, schema, scoring, baselines, held-out test split, and leakage checks. Locations: `benchmark/retrace_bench/`, `data/retrace_bench/`, `data/retrace_supervision/`, `docs/retrace_bench/`, benchmark scripts/tests.
2. **ReTrace-Learn** — method track. Owns the pipeline Graph Extractor -> Typed Revision Proposer -> Authorization Court. Locations: `src/retrace_learn/`, `src/retracemem/`, method scripts/docs.

**ReTrace-Engine** is the implementation name for the deterministic Authorization Court **inside ReTrace-Learn** — it is not a standalone paper, a standalone top-level track, or a third module.

Out of active scope (backlog only, do not add code/active docs): ReTrace-SkillOpt / frozen-agent skill optimization, `memory_policy.md` optimization, Microsoft SkillOpt integration.

## One-Sentence Alignment

ReTrace preserves immutable evidence and changes a belief's eligibility for current answers only through verified, temporally valid typed defeat paths computed by deterministic DPA.

## Method Boundary / Identity

Paper 1 is centered on multi-agent/subagent shared-memory revision authorization.

Multiple subagents may submit evidence-bearing memory updates to a shared long-term memory. ReTrace controls which revisions are allowed to affect the shared usable memory basis.

Stage naming and configuration hierarchy:
- Prompt-Proposer / Stage A = `ReTrace-Prompt` (API baseline model proposes typed revision actions over a fixed candidate view, then routes through ReTrace-Engine).
- DirectJudge / Stage B = `DirectJudge-API` (API baseline model directly predicts final belief usability status, completely bypassing the ReTrace-Engine).
- ReTrace-Learn = `ReTrace-Learn` (the main trainable system: consumes raw dialogues/submissions -> extracts graph nodes/dependencies via a learned Graph Extractor -> proposes typed actions via a learned Typed Revision Proposer -> commits through deterministic ReTrace-Engine).

Public API Boundaries:
- `authorize(...)` is the public deterministic authorization kernel inside ReTrace-Engine. Neither Defeat-Path Authorization (DPA) nor RevisionGate should be invoked directly by external callers. All updates/admissions and deterministic routing happen entirely inside `authorize`.
- `commit_subagent_submission(...)` and `commit_submission_sequence(...)` are multi-agent integration wrappers around `authorize(...)`.

STALE/CUPMem is an external validation/baseline pathway, not the definition of the paper.

Latent memory, RL consolidation, and delayed-utility learning belong to Paper 2.

## Paper 1 ReTrace-Learn Training Boundary

Paper 1 includes ReTrace-Learn: learning an explicit typed revision proposal policy
for multi-agent/subagent shared-memory updates.

The learned policy consumes only method-visible inputs:
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
ReTrace-Learn proposal
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
- `references/` is the tracked lightweight reference registry/notes directory.
  Store only YAML pointers and short Markdown notes there.
- `reference/` is local-only for cloned external repositories or downloaded
  papers. Do not edit it as source, and never commit it.
- Do not commit `reference/`, `outputs/`, `artifacts/`, `analysis/`, caches,
  local environments, generated artifacts, benchmark downloads, model
  checkpoints/weights, or API keys.
- After running scripts, tests, builds, or imports, proactively look for and
  remove local cache/generated directories before committing. At minimum check
  for `.pycache_compile/`, `.pytest_cache/`, `__pycache__/`, `*.pyc`,
  `*.egg-info/`, `.DS_Store`, `outputs/`, `artifacts/`, `analysis/`,
  `data_external/`, `reference/`, `models/`, `checkpoints/`, `adapters/`,
  `wandb/`, and `runs/`.
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

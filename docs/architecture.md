# Architecture (ReTrace-Learn method track)

This document describes the internal architecture of the **ReTrace-Learn** method
track. (For the umbrella project's two-track structure see
[`project_governance.md`](project_governance.md).)

Within ReTrace-Learn, the **two learned stages** (Graph Builder + Proposal
Policy) are separated from the **deterministic commit path**, implemented by
**ReTrace-Engine**. ReTrace-Engine is an *implementation detail* of ReTrace-Learn
(not a standalone track or paper-level module): it never calls an LLM API and is
fully deterministic; everything model- or benchmark-specific lives outside it.
DPA-guided RSFT/DPO is the training *protocol* layered on top — DPA itself does
not learn.

## Layers

```text
ReTrace-Learn (Graph Builder + Proposal Policy) ── learned stages
        │  proposes closed-world typed revision actions
        ▼
ReTrace-Engine Backend (deterministic authorization — implementation detail)
    ├── Parser + RevisionGate  ── structural, local, auditable admission of proposed edges
    ├── authorize(view, proposal_batches, …)  ── the single public kernel (runs DPA)
    └── Defeat-Path Authorization (DPA)  ── computes final statuses
        ▼
SharedMemoryCommitResult  ── authorized snapshot + audit trace
```

### Deterministic Backend: ReTrace-Engine (`src/retracemem/`)

- `schemas.py` — immutable dataclass contracts: `EvidenceNode`, `BeliefNode`, `ConditionNode`, `DependencyEdge(REQUIRES)`, and the `EvidenceEdge` types (`BLOCKS`, `RELEASES`, `SUPERSEDES`, `REAFFIRMS`, `UNCERTAIN`).
- Authorization / TMS — the Defeat-Path Authorization algorithm (DPA) and the `authorize(...)` kernel. DPA assigns each candidate belief a final status in `{AUTHORIZED, BLOCKED, SUPERSEDED, UNRESOLVED}` under the canonical precedence `SUPERSEDES > PREREQUISITE_BLOCK > UNRESOLVED_UNCERTAIN > AUTHORIZED`, with deterministic temporal tie-breaking.
- `multiagent/` — `commit_subagent_submission(...)` and `commit_submission_sequence(...)`: thin wrappers that order subagent submissions deterministically and route them through `authorize(...)`.

### Proposers & Learned Scaffolding: ReTrace-Learn (`src/retracemem/proposers/` & `src/retrace_learn/`)

- `typed_revision_policy.py` — `PromptTypedRevisionPolicy` and the `ClosedAPIZeroShot*` proposers used by **Prompt-Proposer (Stage A)**. Builds prompts from method-visible candidate structure only.
- `replay.py` — the replay path: parses decoded generations into typed actions, with optional constrained post-validation.
- `src/retrace_learn/` — the learned Graph Builder and Proposal Policy stages, rollouts, SFT datasets, and the DPA-guided RSFT/DPO training signal (online GRPO is a future/optional extension).

### Shared evaluation engine (`src/retracemem/evaluation/multiagent/`)

| Module | Responsibility |
| --- | --- |
| `config.py` | `EvalRunConfig`, live API client construction |
| `cases.py` | load fixed-candidate eval cases; namespace renaming |
| `pipeline.py` | `run_retrace_variant_on_episode`: typed proposal → commit → DPA (Stage A/C) |
| `directjudge.py` | Stage B prompt/parse + per-episode DirectJudge run |
| `metrics.py` | pure metric computation (action metrics, grounding, status accuracy) — no I/O |
| `artifacts.py` | structured run-output writing (JSONL / CSV / manifest) |
| `runner.py` | Stage A vs Stage B orchestration (`run_stageab_eval`) + CLI `main` |
| `stagec.py` | Stage C orchestration (`run_stagec_eval`) |
| `data/` | dev-set builders and the Stage C dataset exporter |

## Structured Error Contracts (`retrace_learn.runtime.engine_errors`)

Every stage of the deterministic backend reports failures and warnings through
a shared `EngineError` frozen dataclass. This enables:

- **Structured reward shaping**: penalties proportional to error severity
  (gate rejection penalty, parser error penalty).
- **Auditable fail-closed behavior**: every rejection carries a machine-readable
  stage + code so downstream analysis can categorize failures without parsing prose.
- **Curriculum-driven training**: errors are categorized for analysis
  (`PARSER_ERROR`, `GATE_REJECTION`, `STALE_PROPAGATION`, etc.).

```text
EngineError(stage, code, message, severity, fail_closed, action_index, belief_id, ...)
  ├── stage: PARSER | REVISION_GATE | DPA
  ├── severity: INFO | WARNING | ERROR
  └── fail_closed: True → the action was rejected and produced no graph mutation
```

Canonical error code constants are defined per stage (e.g. `PARSER_INVALID_JSON`,
`GATE_UNKNOWN_BELIEF`, `DPA_MISSING_EVIDENCE_ATOM`).

`ParseResult.errors` carries parser-stage errors. `RuntimeResult.engine_errors`
aggregates errors from all three stages. `compute_reward()` consumes engine
errors to derive `gate_rejection_penalty` and `no_revision_overuse_penalty`.

## Invariants

- `authorize(...)` is the **only** public authorization entrypoint; DPA and
  `RevisionGate` are never called directly by external callers.
- The core is API-free and deterministic; the same inputs always produce the
  same authorized snapshot and trace.
- Proposers see only method-visible inputs (candidate beliefs/replacements,
  conditions, pre-existing `REQUIRES` anchors, the evidence-bearing submission).
  They never see typed gold targets or evaluator final statuses.
- Every typed action must cite the visible new evidence that grounds it.
- The active package (`src/retracemem/`) never imports `experiments.*` or any
  `run_*` script — enforced by `tests/test_active_package_boundary.py`.

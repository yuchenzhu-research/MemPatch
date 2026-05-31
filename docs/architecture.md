# Architecture

ReTrace separates a **deterministic authorization core** from **proposer
families** and **evaluation glue**. The core never calls an API and is fully
deterministic; everything model- or benchmark-specific lives outside it.

## Layers

```text
proposer (Stage A prompt / Stage C adaptive)        ── model-facing, replaceable
        │  typed revision actions over candidate structure
        ▼
RevisionGate  ── structural, local, auditable admission of proposed edges
        ▼
authorize(view, proposal_batches, …)  ── the single public kernel
        │  (internally runs deterministic DPA; external callers never call DPA directly)
        ▼
SharedMemoryCommitResult  ── authorized snapshot + audit trace
```

### Deterministic core (`src/retracemem/`)

- `schemas.py` — immutable dataclass contracts: `EvidenceNode`, `BeliefNode`,
  `ConditionNode`, `DependencyEdge(REQUIRES)`, and the `EvidenceEdge` types
  (`BLOCKS`, `RELEASES`, `SUPERSEDES`, `REAFFIRMS`, `UNCERTAIN`).
- Authorization / TMS — the Defeat-Path Authorization algorithm (DPA) and the
  `authorize(...)` kernel. DPA assigns each candidate belief a final status in
  `{AUTHORIZED, BLOCKED, SUPERSEDED, UNRESOLVED}` under the canonical precedence
  `SUPERSEDES > PREREQUISITE_BLOCK > UNRESOLVED_UNCERTAIN > AUTHORIZED`, with
  deterministic temporal tie-breaking.
- `multiagent/` — `commit_subagent_submission(...)` and
  `commit_submission_sequence(...)`: thin wrappers that order subagent
  submissions deterministically and route them through `authorize(...)`.

### Proposers (`src/retracemem/proposers/`)

- `typed_revision_policy.py` — `PromptTypedRevisionPolicy` and the
  `ClosedAPIZeroShot*` proposers used by **Stage A** (and reused as the
  API-ZeroShot member of the Stage C family). Builds prompts from
  **method-visible** candidate structure only.
- `replay.py` — the **Stage C** adaptive-proposer replay path: parses decoded
  adapter/SFT/ICL generations into typed actions, with optional constrained
  post-validation against the candidate affordances.

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

# Paper 1 Positioning

## One sentence

ReTrace preserves immutable evidence and changes a belief's eligibility for
current answers **only** through verified, temporally valid typed defeat paths
computed by a deterministic Defeat-Path Authorization kernel.

## What Paper 1 is about

**Multi-agent / subagent shared-memory revision authorization.** Multiple subagents submit evidence-bearing memory updates to a shared long-term memory. The primary contribution is a trainable shared-memory framework (**ReTrace-Learn**) that learns to extract memory graphs and propose structured typed actions under the feedback of a deterministic verification backend (**ReTrace-Engine**).

The contribution is the **verifiable revision learning system** plus a fair method comparison (Prompt-Proposer vs. DirectJudge baseline vs. ReTrace-Learn) over identical contexts, with a deterministic authorization engine guaranteeing reproducible commits and training signals.

## In scope

- The deterministic `authorize(...)` kernel inside ReTrace-Engine, DPA semantics, and precedence.
- `RevisionGate` structural / local / auditable admission.
- The minimal, expressive typed action vocabulary (`SUPERSEDES`, `BLOCKS`, `RELEASES`, `REAFFIRMS`, `UNCERTAIN`, `NO_REVISION`) and its distinction from DPA **final statuses** (`AUTHORIZED`, `BLOCKED`, `SUPERSEDED`, `UNRESOLVED`).
- Evidence provenance / grounding requirements (such as the optional `scope` field).
- ReTrace-Learn SFT / RSFT / DPO proposal policy optimization, where the final authorization stays deterministic and API-free.

## Out of scope (belongs to Paper 2)

- Latent / hidden memory representations and hidden-state consolidation.
- Learned forgetting.
- RL over memory state.
- Long-horizon delayed-future-utility learning.
- Biological-memory mechanisms.

These must not be introduced into Paper 1 code or docs. Paper 1 may later test
short-horizon **explicit-action** refinement only if it introduces no latent
memory or hidden-state consolidation.

## Boundaries to respect

- STALE/CUPMem is an **external validation / baseline** pathway, not the
  definition of the method. External bridge code must not redefine the primary
  method identity or the main evaluation data model. STALE adapters/runners stay
  strictly inside `experiments/` (now `experiments/archive/`).
- Do not turn the repo into a generic orchestration framework, RAG system, or a
  Mem0/Graphiti clone; do not implement agent debate/voting; do not duplicate
  the `authorize(...)` kernel; do not change DPA semantics absent a demonstrated
  deterministic bug; do not leak gold fields into method inputs.

# Research positioning (ICLR Paper 1)

> This is the short ICLR-facing framing. The detailed in-scope/out-of-scope
> contract lives in [`paper1_positioning.md`](paper1_positioning.md) and the
> binding rules in `AGENTS.md`.

## Problem

**Multi-agent / subagent shared-memory revision authorization.** When several
subagents write evidence-bearing memory updates into a shared long-term memory,
which revisions are allowed to change the shared *usable* memory basis?

This is **not** retrieval, **not** agent debate/voting, **not** generic
orchestration, and **not** a memory store. ReTrace is the **authorization
mechanism** that sits between proposed revisions and the committed snapshot.

## Claim

The LLM is **not** the memory authority. A proposer emits **typed revision
actions**; a structural `RevisionGate` admits or rejects them; a deterministic
Defeat-Path Authorization (DPA) kernel assigns each belief a final status with
fixed precedence. Immutable evidence is never deleted — only a belief's
*eligibility* to answer the current query changes, through verified, temporally
valid typed defeat paths.

```text
A_t(b) = DPA(b, S_t) ∈ {AUTHORIZED, BLOCKED, SUPERSEDED, UNRESOLVED}
precedence:  SUPERSEDES > PREREQUISITE_BLOCK > UNRESOLVED_UNCERTAIN > AUTHORIZED
```

Typed **actions** (`SUPERSEDES`, `BLOCKS`, `RELEASES`, `REAFFIRMS`, `UNCERTAIN`,
`NO_REVISION`) are distinct from DPA **final statuses** — conflating them is a
category error the parser rejects (a final status used as an action fails
closed).

## Experimental decomposition

* **Stage A (`ReTrace-API-ZeroShot`)** tests whether *typed decomposition +
  deterministic authorization* beats…
* **Stage B (`DirectJudge-API`)**, a direct final-status judge with no typed
  actions / gate / DPA.
* **Stage C (`ReTrace-AdaptiveProposer`)** tests whether an adaptive typed
  proposer improves proposal quality while the gate and DPA stay deterministic
  and unchanged.

All stages are compared over **identical** fixed-candidate contexts (fair
comparison); the commit path is shared and append-only.

## Reproducibility

Every run emits a manifest binding the result to a git commit, provider, model,
temperature, prompt-template hash, parser version, and data split. The kernel is
deterministic, so authorized snapshots are reproducible from the trace.

## Next scientific milestones

Action ablation (which typed actions carry the gains) and multi-action
composition, plus a live Stage A-vs-B failure attribution. Latent memory, RL
consolidation, learned forgetting, and delayed-future-utility learning are
**Paper 2** and must not enter Paper 1.

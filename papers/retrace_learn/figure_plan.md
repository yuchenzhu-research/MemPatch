# ReTrace-Learn — Figure Plan

## Figure 1 — Method Pipeline

- **Purpose:** show raw subagent submissions flowing through Graph Extractor,
  Typed Revision Proposer, and Authorization Court.
- **Input:** `docs/retrace_learn_pipeline.md`, `docs/architecture.md`.
- **Producible now:** yes.

## Figure 2 — Typed Actions Vs. Final Status

- **Purpose:** distinguish proposal actions (`SUPERSEDES`, `BLOCKS`, `RELEASES`,
  `REAFFIRMS`, `UNCERTAIN`, `NO_REVISION`) from DPA final statuses
  (`AUTHORIZED`, `BLOCKED`, `SUPERSEDED`, `UNRESOLVED`).
- **Input:** `src/retracemem/schemas.py`, `src/retracemem/authorization.py`.
- **Producible now:** yes.

## Figure 3 — Experiment Matrix

- **Purpose:** show Stage A / Stage B / Stage C comparisons across E0-E4.
- **Input:** `docs/experiment_protocol.md`.
- **Producible now:** yes.

## Figure 4 — Audit Trace Anatomy

- **Purpose:** show parser, gate, admitted edges, DPA path, and final status for
  one representative submission.
- **Input:** a small smoke case from `scripts/evaluate.py --mock`.
- **Producible now:** yes.

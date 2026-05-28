# ReTrace Project Logic

This document is the alignment source for every future coding agent. Do not
reinterpret the project as generic RAG, a Mem0 clone, a Graphiti clone, a
CUPMem fixed-slot clone, or an RL memory manager.

## Research Direction

ReTrace studies evidence-preserving reversible belief revision for dynamic
agent memory.

The core question is:

> When later evidence weakens or defeats an earlier user belief, can the agent
> stop using the earlier belief as a current premise while preserving the
> original evidence and making the revision path auditable?

The implementation should express this distinction:

- Evidence is historical and immutable.
- Beliefs are open-text propositions derived from evidence.
- Current use of a belief is authorized or blocked by explicit local relations.
- A blocked belief is not deleted.
- Later evidence can restore or alter authorization.

## What ReTrace Is

ReTrace is a conservative memory authorization layer.

It has these components:

```text
incoming session/evidence
→ EpisodeLedger (EvidenceNode)
→ BeliefStore (BeliefNode, ConditionNode, DependencyEdge, EvidenceEdge)
→ candidate prior beliefs
→ RequirementInducer & EvidenceEdgeVerifier
→ RevisionGate
→ DefeatPathAuthorizationAlgorithm (DPA)
→ Query-conditioned basis (BasisBuilder)
→ answer model
```

## What ReTrace Is Not

ReTrace is not:

- a normal retrieval-only memory system;
- a destructive consolidation system;
- a fixed ontology state tracker;
- a graph database benchmark;
- a latent memory learner;
- an RL/GRPO memory action policy;
- a new benchmark paper;
- a hand-written rule engine or heuristic pipeline.

## Methodology & Contributions

ReTrace centers on three paper contributions:
1. **Reversible authorization formulation**: Dynamic memory revision changes which evidence-grounded beliefs may govern current answers, without deleting historical episodic evidence.
2. **Defeat-Path Authorization Algorithm**: A belief is blocked or superseded only through admitted typed paths over REQUIRES, BLOCKS, RELEASES, SUPERSEDES, REAFFIRMS, and UNCERTAIN edges.
3. **Structural attribution evaluation**: Compare `ReTrace-LLM` against a matched `DirectJudge-LLM` baseline and external memory systems, while treating heuristic verifiers only as development fixtures.

Target implementations:
* **`ReTrace-LLM` (Main method)**: Generic typed-edge prediction by LLM + deterministic DPA.
* **`DirectJudge-LLM` (Attribution baseline)**: Same model and similar context/call budget deciding final adjudication directly without DPA or local edge restrictions.
* **`ReTrace-Local` (Enhancement)**: Optional later learned local verifier + deterministic DPA.
* **Heuristics (Dev-only)**: `HeuristicRequirementInducer` and `HeuristicEvidenceEdgeVerifier` are development-only deterministic fixtures. They must not be used as the paper's main method.

## Difference From Key References

### CUPMem / STALE

CUPMem is the closest method reference for STALE-style stale memory handling.
It uses typed state tracks and write-time adjudication. ReTrace should use STALE
as a benchmark and CUPMem as a baseline, but ReTrace core should not inherit
fixed life-domain slots as its primary representation.

ReTrace differs by:

- preserving original episodic evidence;
- using open-text belief nodes;
- requiring a traceable defeat path;
- keeping authorization reversible.

### Memora

Memora is a primary benchmark for repeated mutation and obsolete-memory misuse.
It is not a method template. ReTrace should adapt to Memora's memory-system
interface and FAMA evaluation shape.

### Mem0

Mem0 is a useful engineering baseline and API-shape reference. ReTrace should
not become Mem0 with different prompts. If wrapping Mem0 later, add explicit
metadata for `episode_id`, `source_message_ids`, and `run_step_id`.

### NEMORI and TriMem

Use NEMORI and TriMem for provenance ideas:

- episode-first memory creation;
- source message/dialogue IDs;
- semantic/fact nodes linked back to episodes.

Do not turn ReTrace into a distillation method.

### Graphiti

Use Graphiti for temporal provenance and valid/invalid time ideas. Do not make
Paper 1 depend on a graph database or open-world temporal KG discovery.

## Revision Semantics & DPA Rules

### 1. Dependency Graph (`DependencyEdge`)
*   `REQUIRES`: belief --REQUIRES--> condition. Identifies prerequisites for using a belief.

### 2. Evidence Updates (`EvidenceEdge`)
*   `BLOCKS`: evidence --BLOCKS--> condition. Defeats a prerequisite.
*   `RELEASES`: evidence --RELEASES--> condition. Clears a blocker. RELEASES removes an active blocker and may restore eligibility; it never itself reasserts current truth and never overrides a later SUPERSEDES edge.
*   `SUPERSEDES`: evidence --SUPERSEDES--> belief. Replaces the prior belief. Requires a real `replacement_belief_id` from candidate beliefs extracted from the new evidence.
*   `REAFFIRMS`: evidence --REAFFIRMS--> belief. Clears belief-level uncertainty.
*   `UNCERTAIN`: evidence --UNCERTAIN--> belief. Signals information is insufficient, removing default authorization.

### 3. Core Gate and DPA rules
- **No mock-ledger fallback**: Decisions are purely structural on the verified-edge graph.
- **Evidence preservation**: Episodes are append-only and never deleted.
- **Path-conditioned**: `BLOCKS` and `RELEASES` only affect authorization when they target a condition node linked via `REQUIRES` to a candidate belief.
- **Supersession Grounding**: A `SUPERSEDES` edge must link to a real replacement belief grounded in the new evidence.
- **Reversibility**: Releasing a condition blocker or reaffirming an uncertain belief restores its candidate status.

## First-Version Success Criteria

The first research code version is done when:

1. A small BoundaryAudit set runs end to end under `ReTrace-LLM` and DPA.
2. STALE and Memora smoke loaders run without API keys.
3. `DirectJudge-LLM` and `ReTrace-LLM` run on identical evaluation budgets.
4. Core TMS and verifier contract cases are covered by tests.
5. Heuristic verifiers remain isolated strictly for contract verification.
6. The code is easy for later agents to extend without changing research scope.


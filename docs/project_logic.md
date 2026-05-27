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
→ EpisodeLedger
→ BeliefStore
→ candidate prior beliefs
→ RelationVerifier
→ RevisionGate
→ AuthorizationEngine
→ BasisBuilder
→ answer model
```

The first version must stay small and deterministic enough to run without API
keys.

## What ReTrace Is Not

ReTrace is not:

- a normal retrieval-only memory system;
- a destructive consolidation system;
- a fixed ontology state tracker;
- a graph database benchmark;
- a latent memory learner;
- an RL/GRPO memory action policy;
- a new benchmark paper.

Do not add these directions unless a later plan explicitly changes the scope.

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

## Revision Semantics

Supported local relation labels:

- `SUPPORT`: later evidence continues to support the belief.
- `SUPERSEDE`: later evidence replaces an earlier belief.
- `BLOCK`: later evidence defeats a prerequisite for current use.
- `CONDITION`: belief remains true but use requires a condition.
- `NONE`: no authorized revision.
- `UNCERTAIN`: information is insufficient; do not use the prior belief as a
  current default.
- `REQUIRED_BY`: prerequisite relation used by later two-hop versions.

First-version gate rules:

- `NONE` must never revise a belief.
- `SUPERSEDE` can block the old belief only when it points to a replacement or
  target belief.
- `BLOCK` must have a condition or explicit target/prerequisite.
- `UNCERTAIN` can remove current default authorization but must not invent a new
  belief.
- Unrelated beliefs must remain authorized.

## First-Version Success Criteria

The first research code version is done when:

1. A small BoundaryAudit set runs end to end.
2. STALE and Memora smoke loaders run without API keys.
3. Retrieval baseline emits unified JSONL.
4. ReTrace heuristic pipeline emits unified JSONL.
5. Core TMS cases are covered by tests.
6. The code is easy for later agents to extend without changing research scope.


# ReTrace Code Direction

This repository should start as an integration and method scaffold around the
ICLR 2027 Paper 1 plan:

> Evidence-preserving reversible belief revision for evolving personalized
> agents.

The first implementation target is not full latent memory, RL memory actions,
or a new benchmark. The first target is a small reproducible loop:

1. Load STALE and Memora examples through adapters.
2. Run a retrieval-only baseline with a shared JSONL output schema.
3. Run prompt-based ReTrace relation verification.
4. Apply a conservative TMS-inspired revision gate.
5. Build an authorized current basis for a fixed answer model.
6. Report official benchmark metrics and cost/call counters.

## Reference Repositories

External code is cloned under `reference/` and should remain isolated from the
project implementation.

### Primary references to run

- `reference/STALE`
  - Use for STALE data format, official evaluation, and CUPMem baseline.
  - Important areas:
    - `STALE/Evaluation/`
    - `cup_mem/query/`
    - `cup_mem/write/`
    - `cup_mem/store_layer/`
  - ReTrace should borrow the evaluation interface and readout discipline, but
    not CUPMem's fixed typed state ontology as the core memory representation.

- `reference/Memora`
  - Use for Memora temporal mutation settings, FAMA evaluation, task splits, and
    existing memory method integration points.
  - ReTrace should use Memora as a main benchmark, not as a method template.

### References to absorb selectively

- `reference/nemori`
  - Useful for episode integration, distillation interfaces, and benchmark
    runner structure.
  - It is a baseline or comparison point, not the ReTrace architecture.

- `reference/graphiti`
  - Useful for temporal provenance and graph storage design.
  - ReTrace should not become a general temporal KG system in Paper 1.

- `reference/TriMem`
  - Useful for raw dialogue IDs, atomic facts, profiles, and provenance links.

- `reference/A-mem-sys` and `reference/mem0`
  - Useful as engineering memory baselines or wrappers.

### References for later or related work

- `reference/MemoryAgentBench` and `reference/LongMemEval`
  - Supplemental/non-regression benchmarks after STALE and Memora are working.

- `reference/Adaptive_Memory_Admission_Control_LLM_Agents`
  - Admission-control baseline. It does not solve revision authorization.

- `reference/AgeMem`, `reference/MEM1`, `reference/OpenTinker`, `reference/verl`
  - Related work or Paper 2 material. Do not block Paper 1 on these.

## First-Stage Module Ownership

The local implementation should be owned by these packages:

- `retracemem.memory`
  - Immutable episodic evidence ledger.
  - Open-text belief store.
  - Temporal validity helpers.

- `retracemem.verifier`
  - Relation labels and verifier interface.
  - Prompt verifier first.
  - SFT data generation and trained verifier later.

- `retracemem.tms`
  - Conservative authorization gate.
  - Direct `SUPERSEDE` and two-hop `BLOCK -> REQUIRED_BY` paths only.
  - Rollback when blockers expire or are defeated.

- `retracemem.generation`
  - Query-time authorized basis builder.
  - Fixed answer model wrapper.

- `retracemem.adapters`
  - STALE and Memora adapters first.
  - MemoryAgentBench and LongMemEval later.

- `retracemem.baselines`
  - Retrieval-only baseline.
  - CUPMem and NEMORI wrappers.
  - Direct LLM revision baseline.

- `retracemem.evaluation`
  - Unified JSONL output.
  - Official metric wrappers.
  - Cost, token, call, and latency tracking.

## Unified JSONL Output

All methods should emit records shaped like:

```json
{
  "query_id": "...",
  "method": "retrace_full",
  "retrieved_evidence": [],
  "candidate_beliefs": [],
  "authorized_basis": [],
  "blocked_beliefs": [],
  "answer": "...",
  "tokens": {},
  "calls": {},
  "latency_ms": 0
}
```

This is the integration contract between adapters, methods, and evaluation.

## MVP Order

1. Implement local schemas and pure Python TMS gate tests.
2. Add STALE adapter that can read a small official/example split.
3. Add Memora adapter that can read a small official/example split.
4. Add retrieval-only baseline and unified JSONL writer.
5. Add prompt verifier and ReTrace full method for hand-written cases.
6. Run STALE small.
7. Run Memora small.
8. Generate RevisionPairs-Train and start learned verifier only after the prompt
   interface is validated.

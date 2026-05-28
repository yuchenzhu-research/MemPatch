> [!IMPORTANT]
> This document is retained solely as the initial planning reference for the flat relation prototype.
> The implementation has transitioned to a typed graph + Defeat-Path Authorization (DPA) core. All runtime specifications are governed by [docs/refactor_plan_defeat_path.md](file:///Users/yuchenzhu/Desktop/ReTrace/docs/refactor_plan_defeat_path.md).

# ReTrace Implementation Plan

This plan turns the paper blueprint into a first-stage codebase. The goal is a
small, reproducible method loop for Paper 1, not a full benchmark suite, RL
memory manager, or latent memory system.

## 1. Scope Lock

Paper 1 implementation is:

```text
immutable episodic evidence
→ open-text belief nodes
→ candidate affected-belief retrieval
→ local relation verification
→ conservative TMS-inspired authorization gate
→ query-time authorized basis
→ fixed answer model
```

Out of scope for this stage:

- Full latent memory or learned semantic consolidation.
- RL/GRPO memory action training.
- New benchmark construction beyond a small BoundaryAudit diagnostic set.
- Open-world causal topology discovery.
- Direct adoption of CUPMem's fixed typed slot ontology as ReTrace's core.

## 2. Reference Code Policy

All external code stays under `reference/`. Local ReTrace code should wrap or
adapt reference repositories instead of modifying them.

Primary benchmark references:

- `reference/STALE`
  - Run and wrap STALE/CUPMem.
  - Reuse evaluation entry points and metric logic.
  - Do not copy fixed ontology assumptions into ReTrace.

- `reference/Memora`
  - Run and wrap Memora FAMA evaluation.
  - Align local backend interface with Memora's memory-system shape.

Selective design references:

- `reference/nemori`: episode integration and baseline wrapper.
- `reference/graphiti`: temporal provenance and graph-storage ideas.
- `reference/TriMem`: source dialogue IDs, atomic facts, profile layering.
- `reference/A-mem-sys` and `reference/mem0`: engineering baselines.

Later or related-work references:

- `reference/MemoryAgentBench`
- `reference/LongMemEval`
- `reference/AgeMem`
- `reference/MEM1`
- `reference/OpenTinker`
- `reference/verl`
- `reference/Adaptive_Memory_Admission_Control_LLM_Agents`

## 3. Local Package Layout

Target package:

```text
retracemem/
  adapters/
    stale_adapter.py
    memora_adapter.py
  backends/
    base.py
    retrace_backend.py
    retrieval_baseline.py
    cupmem_wrapper.py
    mem0_wrapper.py
    nemori_wrapper.py
  evaluation/
    jsonl.py
    cost_tracker.py
    stale_metrics.py
    memora_fama.py
    runner.py
  generation/
    basis_builder.py
    answerer.py
  memory/
    episode_ledger.py
    belief_store.py
    belief_extractor.py
    temporal_validity.py
  retrieval/
    candidate_retriever.py
  tms/
    gate.py
    authorization.py
    rollback.py
  verifier/
    base.py
    prompt_verifier.py
    sft_data.py
  schemas.py
scripts/
  run_stale.py
  run_memora.py
  run_boundary_audit.py
configs/
  retrace_prompt.yaml
  stale.yaml
  memora.yaml
tests/
```

## 4. Core Data Contracts

The stable data contracts live in `retracemem/schemas.py`.

Minimum objects:

- `EpisodicEvidence`
  - `id`, `timestamp`, `text`, `source_id`, `metadata`
  - append-only; never overwritten by memory revision.

- `Belief`
  - `id`, `proposition`, `supported_by`, `status`, `metadata`
  - open-text proposition; no fixed life-domain slot required.

- `RelationPrediction`
  - `SUPPORT`, `SUPERSEDE`, `BLOCK`, `CONDITION`, `NONE`, `UNCERTAIN`,
    `REQUIRED_BY`
  - includes evidence span, rationale, confidence, optional temporal validity.

- `AuthorizationDecision`
  - `belief_id`, `authorized`, `reason`, `justification_path`

- `EvaluationRecord`
  - unified method output for all benchmark runs.

Unified JSONL output:

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

## 5. First Milestone: Pure Local ReTrace Logic

Implement and test without API calls:

1. `EpisodeLedger`
   - append-only evidence storage.
   - duplicate IDs raise an error.

2. `BeliefStore`
   - open-text belief storage.
   - relation storage and lookup.

3. `RevisionGate`
   - accepts direct `SUPERSEDE`.
   - accepts `BLOCK` only when a condition or explicit prerequisite exists.
   - treats `UNCERTAIN` as "not authorized as current default" without inventing
     a replacement.
   - rejects `NONE` as a revision operation.

4. `AuthorizationEngine`
   - returns blocked for valid `BLOCK`.
   - returns superseded for valid `SUPERSEDE`.
   - returns not authorized for `UNCERTAIN`.
   - preserves unrelated beliefs.

5. `BasisBuilder`
   - emits only authorized beliefs.
   - later includes provenance and blocking evidence.

Acceptance tests:

- A broken-leg event blocks a bicycle-commute belief.
- The same broken-leg event does not block an unrelated food preference.
- A superseding address belief blocks the old address belief.
- An uncertain relation removes the old belief as a current default but does not
  create a new belief.

## 6. Second Milestone: Benchmark-Neutral Backend Interface

Local backend interface:

```python
reset_user(user_id)
ingest_session(user_id, session, metadata=None)
search(user_id, query, limit=10, metadata=None)
answer(user_id, query, retrieved, metadata=None)
```

This follows Memora's memory-system style while still supporting STALE's sample
runner pattern.

Implement:

- `ReTraceBackend`
  - owns `EpisodeLedger` and `BeliefStore` per user.
  - later wires belief extraction, candidate retrieval, verifier, and gate.

- `RetrievalBaselineBackend`
  - stores raw evidence only.
  - returns lexical/substring scored evidence.
  - provides a non-LLM smoke-test baseline for adapters.

Acceptance tests:

- A user can be reset, sessions ingested, searched, and answered.
- Results can be converted to `EvaluationRecord` JSONL.

## 7. Third Milestone: STALE Adapter

Reference paths:

- `reference/STALE/STALE/Evaluation/run_target_model.py`
- `reference/STALE/STALE/Evaluation/full_eval_performance.py`
- `reference/STALE/cup_mem/core/sample_runner.py`
- `reference/STALE/cup_mem/pipeline.py`

Implement `retracemem/adapters/stale_adapter.py`:

- discover available STALE JSON files.
- load `*_MAIN.json` records.
- normalize each record to:
  - `sample_id`
  - chronological `sessions`
  - `timestamps`
  - probing queries for dimensions 1/2/3
  - old/new memory fields
  - explanation and metadata.

Do not implement a full STALE judge yet unless the official API dependencies
are configured. First target is data loading and local smoke runs.

Acceptance tests:

- Adapter can load a small/demo STALE file if present.
- Normalized samples preserve `uid`, `M_old`, `M_new`, `query_time`,
  `haystack_session`, and probing queries.

## 8. Fourth Milestone: Memora Adapter

Reference paths:

- `reference/Memora/data/README.md`
- `reference/Memora/evals/agent_eval/base_evaluator.py`
- `reference/Memora/evals/agent_eval/conversation_to_memory.py`
- `reference/Memora/evals/agent_eval/memory_to_answer.py`
- `reference/Memora/evals/model_eval/aggregate_results.py`

Implement `retracemem/adapters/memora_adapter.py`:

- discover personas and period directories.
- load chronological `session_*.json`.
- load `evaluation_questions_<persona>.json`.
- normalize samples to:
  - `persona_id`
  - `period`
  - chronological sessions
  - task bucket: remembering / reasoning / recommending
  - question text and rubric/evidence fields.

Acceptance tests:

- Adapter discovers data roots.
- Adapter loads sessions in chronological order.
- Adapter exposes evaluation questions with bucket metadata.

## 9. Fifth Milestone: Prompt Relation Verifier

Implement `retracemem/verifier/prompt_verifier.py` as a real prompt-backed
component after local logic is stable.

Expected output:

```json
{
  "relation": "BLOCK",
  "condition": "cycling ability",
  "span": "I broke my leg yesterday...",
  "rationale": "...",
  "confidence": 0.82
}
```

For now, keep a deterministic placeholder or heuristic verifier for tests.

Acceptance tests:

- Parsed verifier JSON maps to `RelationPrediction`.
- Invalid labels fail closed to `UNCERTAIN`.
- Missing condition on `BLOCK` is rejected by `RevisionGate`.

## 10. Sixth Milestone: BoundaryAudit Mini Set

Create 20 local diagnostic cases before running large benchmarks.

Case categories:

- direct supersession.
- prerequisite blocking.
- unrelated protected belief.
- condition, not invalidation.
- uncertain, no invented replacement.

Store later under:

```text
data/boundary_audit/*.jsonl
```

This is an analysis set only, not a benchmark contribution.

## 11. Subagent Execution Plan

Use subagents only on disjoint write scopes.

### Agent A: Core Logic Worker

Owned files:

- `retracemem/memory/*`
- `retracemem/tms/*`
- `retracemem/generation/basis_builder.py`
- `tests/test_*tms*`
- `tests/test_*memory*`

Task:

- harden pure local logic.
- add temporal validity placeholder only if needed by gate tests.
- add tests for protected unrelated beliefs, supersession, and uncertain cases.

### Agent B: Adapter Worker

Owned files:

- `retracemem/adapters/stale_adapter.py`
- `retracemem/adapters/memora_adapter.py`
- `tests/test_*adapter*`

Task:

- inspect reference data formats.
- implement discovery and best-effort loaders.
- make loaders degrade cleanly when benchmark data files are not present.

### Agent C: Backend/Evaluation Worker

Owned files:

- `retracemem/backends/*`
- `retracemem/evaluation/*`
- `tests/test_*backend*`
- `tests/test_*evaluation*`

Task:

- implement retrieval baseline backend.
- improve JSONL writer/reader helpers.
- add cost tracker tests.
- add conversion helper from backend outputs to `EvaluationRecord`.

## 12. Immediate Next Steps

1. Dispatch Agents A, B, and C with the ownership above.
2. Main thread reviews current reference clone status and closes any unfinished
   clone/session state.
3. Main thread adds missing project docs/config stubs only outside the agents'
   write scopes.
4. Wait for agents, review diffs, and resolve overlap manually.
5. Run `python3 -m compileall -q retracemem tests`.
6. If `pytest` is unavailable, report that and use compileall until a local env
   is created.

## 13. Verification Commands

Current no-dependency checks:

```bash
python3 -m compileall -q retracemem tests
```

Preferred test command after installing pytest:

```bash
python3 -m pytest -q
```

If a virtual environment is added later, keep it local to the repo and avoid
depending on global Python packages.

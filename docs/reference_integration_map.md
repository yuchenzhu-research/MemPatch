# Reference Integration Map

This map records what should be borrowed from each cloned repository and where
it should land in the local ReTrace implementation.

## STALE and CUPMem

Local target:

- `retracemem.adapters.stale_adapter`
- `retracemem.backends.cupmem_wrapper`
- `retracemem.evaluation.stale_metrics`

Reference paths:

- `reference/STALE/STALE/Evaluation/run_target_model.py`
  - Prompt construction, target model evaluation loop, token trimming, and
    response metadata.
- `reference/STALE/STALE/Evaluation/full_eval_performance.py`
  - Judge execution and accuracy aggregation.
- `reference/STALE/STALE/Evaluation/judge_prompts.py`
  - Three-dimension judge rubrics.
- `reference/STALE/cup_mem/run_cup_mem.py`
  - Minimal CUPMem run entry point.
- `reference/STALE/cup_mem/pipeline.py`
  - `CupMemEngine` method lifecycle.
- `reference/STALE/cup_mem/core/sample_runner.py`
  - STALE sample-to-memory execution semantics.
- `reference/STALE/cup_mem/memory/models.py`
  - Useful concepts: profile items, session deltas, invalidation proposals, and
    stale archive traces.

Do not copy CUPMem's fixed slot ontology into ReTrace's core. Use CUPMem as a
baseline wrapper and as a source for evaluation discipline.

## Memora

Local target:

- `retracemem.adapters.memora_adapter`
- `retracemem.backends.base`
- `retracemem.evaluation.memora_fama`

Reference paths:

- `reference/Memora/evals/README.md`
  - Evaluation setup and FAMA definition.
- `reference/Memora/evals/agent_eval/base_evaluator.py`
  - Most important backend interface reference. Align ReTrace's backend API
    with `initialize_client`, `add_conversation_to_memory`, `search_memories`,
    and `get_required_env_vars`.
- `reference/Memora/evals/agent_eval/conversation_to_memory.py`
  - Chronological session ingestion.
- `reference/Memora/evals/agent_eval/memory_to_answer.py`
  - Search, answer, judge, and save-results flow.
- `reference/Memora/evals/model_eval/model_based_evaluator.py`
  - Full-context model evaluator and report schema.
- `reference/Memora/evals/model_eval/aggregate_results.py`
  - Aggregation into FAMA tables.
- `reference/Memora/data/README.md`
  - Session and evaluation question formats.

The backend interface should be benchmark-neutral, but Memora's `BaseMemorySystem`
is the practical shape to start from.

## NEMORI

Local target:

- `retracemem.backends.nemori_wrapper`
- Later `retracemem.verifier.sft_data`

Use NEMORI for episode integration, memory output format, and benchmark runner
ideas. Treat it as a baseline and comparison point, not as ReTrace's method.

Specific paths to inspect:

- `reference/nemori/nemori/domain/models.py`
  - `Message -> Episode -> SemanticMemory` structure.
  - `Episode.source_messages` and `SemanticMemory.source_episode_id` are the
    provenance pattern to borrow.
- `reference/nemori/nemori/core/memory_system.py`
  - Buffer-to-episode-to-semantic write pipeline.
- `reference/nemori/nemori/search/unified.py`
  - Episode + semantic hybrid retrieval and RRF-style merge.
- `reference/nemori/nemori/api/facade.py`
  - Candidate wrapper surface for `add_messages`, `flush`, and `search`.

Wrapper status: useful optional backend, but it has heavier Postgres/Qdrant and
async setup requirements, so it should not be in the first MVP loop.

## Graphiti

Local target:

- Later `retracemem.memory.graph_store`
- Later `retracemem.memory.temporal_validity`

Borrow temporal provenance and graph storage ideas. Do not expand Paper 1 into
open-world temporal KG discovery.

Specific paths to inspect:

- `reference/graphiti/graphiti_core/nodes.py`
  - `EpisodicNode` with content, source description, valid time, and edge links.
- `reference/graphiti/graphiti_core/edges.py`
  - `EntityEdge.fact`, `episodes`, `valid_at`, `invalid_at`, and
    `reference_time`.
- `reference/graphiti/graphiti_core/graphiti.py`
  - `add_episode` lifecycle: previous context, extraction, dedupe,
    invalidation, and hydration.
- `reference/graphiti/graphiti_core/search/search_config_recipes.py`
  - Search recipes combining BM25, cosine, BFS, RRF/MMR, cross-encoder,
    distance, and episode mentions.

Wrapper status: possible later if a graph backend is acceptable. For Paper 1
MVP, borrow schema and temporal invalidation patterns instead of adding graph
infrastructure.

## TriMem

Local target:

- Later `retracemem.memory.belief_extractor`
- Later `retracemem.retrieval.candidate_retriever`

Borrow source dialogue IDs, atomic facts, and entity/profile layering.

Specific paths to inspect:

- `reference/TriMem/models/memory_entry.py`
  - `MemoryEntry` shape: lossless restatement, keywords, timestamp, location,
    persons, entities, topic, and `source_dialogue_ids`.
- `reference/TriMem/core/memory_builder.py`
  - Prompting pattern for atomic, de-referenced, source-grounded facts.
- `reference/TriMem/database/dialogue_store.py`
  - Source dialogue lookup with local context windows.
- `reference/TriMem/core/hybrid_retriever.py`
  - Semantic, keyword, and structured retrieval fusion.

Wrapper status: do not directly wrap first. It is more useful as schema and
retrieval design inspiration.

## A-MEM and Mem0

Local target:

- `retracemem.backends.amem_wrapper`
- `retracemem.backends.mem0_wrapper`

Use as engineering baselines where their dependencies are manageable.

Mem0 specific paths:

- `reference/mem0/mem0/memory/main.py`
  - Mature `add`, `get_all`, `search`, `update`, `delete`, and `history` API.
- `reference/mem0/mem0/configs/prompts.py`
  - Additive memory extraction using recent messages and existing memories.
- `reference/mem0/openmemory/api/app/models.py`
  - Memory state, history, and access-log concepts.
- `reference/mem0/mem0/vector_stores/base.py`
  - Vector store abstraction and metadata filters.

Mem0 wrapper status: best direct wrapper candidate. ReTrace must add
`episode_id`, `source_message_ids`, and `run_step_id` metadata so evidence
remains traceable.

A-MEM specific paths:

- `reference/A-mem-sys/agentic_memory/memory_system.py`
  - `MemoryNote` links, tags, access counters, and evolution history.
- `reference/A-mem-sys/agentic_memory/retrievers.py`
  - Linked-neighbor retrieval behavior.

A-MEM wrapper status: do not directly wrap first. Borrow `links`,
`evolution_history`, and enhanced embedding text only.

## MemoryAgentBench and LongMemEval

Local target:

- Later `retracemem.adapters.memoryagentbench_adapter`
- Later `retracemem.adapters.longmemeval_adapter`

Use only after STALE and Memora are running.

## Paper 2 / Related Work Repositories

Keep these out of the Paper 1 critical path:

- `reference/AgeMem`
- `reference/MEM1`
- `reference/OpenTinker`
- `reference/verl`
- `reference/Adaptive_Memory_Admission_Control_LLM_Agents`

They are valuable for related work and future latent/RL memory work, but they
should not define the first implementation milestone.

## Absorption Order

1. Use Mem0's API surface as the main memory backend shape.
2. Use NEMORI and TriMem for episode/source-message provenance.
3. Use Graphiti for temporal fact edges and valid/invalid time boundaries.
4. Use NEMORI, Mem0, and TriMem for vector + text + metadata retrieval fusion.
5. Use A-MEM only for links, evolution history, and enriched embedding text.

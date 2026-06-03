# Baseline Suite v2

To evaluate the difficulty of ReTrace-Bench v2, we establish a suite of baseline models. These baselines represent a variety of memory handling strategies.

## Baseline Descriptions

1. **latest-only (Heuristic)**:
   - *Strategy*: Only reads/retrieves the single most recent event or memory entry matching a keyword or category.
   - *Role*: Establishes a lower performance bound. Highlights sensitivity to stale memory reuse.
   - *Status*: **Implemented in Pass 1 (`latest_only_v2`)**.

2. **retrieve-all / long-context (LLM Baseline)**:
   - *Strategy*: Feed the entire raw event trace and memory snapshot into a long-context LLM window.
   - *Role*: Evaluates if a model can perform implicit memory updating and retrieval when everything is visible.
   - *Status*: **Heuristic placeholder implemented (`retrieve_all_v2`)**.

3. **RAG memory**:
   - *Strategy*: standard vector search retrieval over past events/memories. No explicit update or state-tracking logic.

4. **CRUD memory manager**:
   - *Strategy*: A basic database-style memory manager implementing simple Create, Read, Update, Delete operations based on heuristic keywords.

5. **Mem0-style memory manager**:
   - *Strategy*: Uses an LLM to dynamically extract, update, and persist key-value facts over a sequence of dialog turns.

6. **CUPMem-style state-aware memory**:
   - *Strategy*: A graph-based memory adapter tracking state invalidation and conflicts.

7. **ReTrace-Learn structured method**:
   - *Strategy*: Uses a learned Graph Builder and Proposal Policy routing proposals through the deterministic DPA engine.

8. **oracle memory-state**:
   - *Strategy*: Directly feeds the gold memory status values into the agent. Establishes the upper bound of task performance if memory tracking is perfect.

9. **oracle evidence**:
   - *Strategy*: Directly feeds the gold supporting event/memory records into the agent. Establishes reasoning limits.

10. **human audit subset**:
    - *Strategy*: A gold standard baseline evaluated on the 100-scenario seed set, showing human performance on memory-state tracking and reasoning.

# Baseline Adapter Audit

The paper baseline runner compares models through one benchmark response contract. The adapters in `context_builders.py` transform only public scenario fields before calling the shared five-field JSON prompt builder. They never read `hidden_gold`.

## Naming

`mem0` and `a_mem` are stable internal IDs. Paper-facing names must be **Mem0-style Proxy** and **A-MEM-style Proxy**. They are interface-compatible proxies, not executions of the upstream systems.

## Shared Contract

All public baselines:

- receive `public_scenario_view(scenario)`;
- use the same base model, decoding settings, and output schema;
- receive no LoRA adapter;
- are scored by the same `benchmark.api.evaluate_predictions` call;
- cannot access expected decisions, memory states, evidence IDs, diagnoses, or answers.

## Mem0-style Proxy

Implementation: `_mem0_units`, `_retrieve_units`, and the `mem0` branch of `build_baseline_view`.

1. Each visible initial memory becomes an `initial_memory` unit with its public text, scope, and source event IDs.
2. Each visible event that names `related_memory_ids` becomes an `event_update` unit.
3. Units are ranked by lexical query overlap; the top `rag_top_k` units are exposed.
4. The three most recent visible events are retained as short-term context.
5. The shared prompt requests the benchmark five-field response.

This proxy does not run upstream extraction, consolidation, deduplication, graph maintenance, storage, or retrieval services. It isolates a Mem0-inspired memory-unit interface under the benchmark's controlled local-model protocol.

## A-MEM-style Proxy

Implementation: `_a_mem_notes`, `_retrieve_a_mem_notes`, and the `a_mem` branch of `build_baseline_view`.

1. Each visible event becomes a note.
2. A note links to public `related_memory_ids` and the immediately previous note.
3. Lexical query overlap selects seed notes.
4. One-hop linked notes are added to the retrieved set.
5. The three most recent visible events remain as short-term context.
6. The shared prompt requests the benchmark five-field response.

This proxy does not reproduce upstream attribute generation, autonomous note evolution, embedding retrieval, or memory-management loops. It isolates a linked-note interface that can be evaluated without external services or hidden-gold leakage.

## Required Appendix Disclosure

The appendix should report:

- exact paper-facing proxy names;
- the six steps above for each adapter;
- `rag_top_k` and recent-event count;
- shared model and decoding controls;
- the absence of upstream service calls and hidden-gold access;
- the limitation that proxy results are not claims about official system performance.

## Human and Independent-LLM Review Checklist

1. Confirm every field entering the adapter is present in `public_scenario_view`.
2. Search the adapter and runner for `hidden_gold` and oracle-field access.
3. Compare the transformation above against the cited system's conceptual interface.
4. Mark omitted upstream components explicitly; do not silently treat them as reproduced.
5. Verify that all methods share model weights, temperature, token budget, and scorer.
6. Inspect at least ten rendered prompts per proxy across decisions and domains.
7. Record reviewer/model identity, date, code commit, disagreements, and final resolution outside the anonymous paper.

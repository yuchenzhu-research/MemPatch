# Schema Definitions

ReTrace-Bench utilizes a strongly typed, deterministic schemas defined in `src/retrace_bench/schemas.py`.

## Core Entities

1. **DialogueTurn**: Tracks conversational turns from subagents.
2. **MemoryEntry**: Node in the memory snapshot (either `belief`, `condition`, or `evidence`).
3. **RevisionAction**: Edge-level proposals (e.g. `SUPERSEDES`, `BLOCKS`).
4. **ProbeQuery**: 4 multiple-choice questions checking different dimensions of DPA capabilities.
5. **Scenario**: Container for the complete dialogue history, topology, snapshots, gold statuses, and queries.

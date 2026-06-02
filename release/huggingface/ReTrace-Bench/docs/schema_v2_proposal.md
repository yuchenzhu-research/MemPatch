# Dataset Schema v2 Proposal

This proposal defines the structure of ReTrace-Bench v2. The schema transitions the benchmark from dialogue-focused structures to realistic, event-based logs.

## Main Objects

The schema comprises the following dataclasses and enums implemented in `schemas_v2.py`:

- **ScenarioV2**: The main evaluation case. It bundles event traces, initial memory, hidden timelines, and downstream tasks.
- **EventV2**: A single log event (e.g. user message, agent message, memory read/write, tool call).
- **MemoryEntryV2**: A memory item. Can represent beliefs, file links, preferences, or tool traces.
- **MemoryLifecycleOperationV2**: An atomic lifecycle state transition (e.g. CREATE, UPDATE, SUPERSEDE, BLOCK, UNBLOCK, FORGET, REACTIVATE).
- **HiddenMemoryLifecycleV2**: The hidden gold timeline of memory operations.
- **TaskV2**: A downstream evaluation task (e.g. Black-box, Memory State, Structured Revision, Oracle Diagnostic).
- **GoldBehaviorV2**: The hidden gold answer, memory statuses, or structured actions.
- **GoldEvidenceV2**: Grounding justification containing supporting events or memories.
- **StructuredRevisionActionV2**: Optional diagnostic DPA-vocabulary revision proposal.
- **ScenarioMetadataV2**: Meta information (annotators, license, contamination policy).
- **ManifestV2**: Top-level manifest mapping splits and versions.
- **PredictionV2**: Participant submission format.
- **EvaluationResultV2**: Metrics outcome for a task.

---

## Migration from v1 to v2

ReTrace-Bench v2 maintains backward compatibility with v1 datasets via a deterministic mapping layer:

| v1 Field | v2 Mapping | Description |
| :--- | :--- | :--- |
| `dialogue_history` | `event_trace` | Each dialogue turn is mapped to a v2 `Event` with type `user_message` or `agent_message`. |
| `probe_queries` | `tasks` | Each query is converted into a `Task` of type `black_box_task` with options formatted inside the prompt. |
| `revision_family` | `hidden_memory_lifecycle` | The implicit revisions in v1 are exported as explicit `MemoryLifecycleOperation` operations. |
| `gold_revision_actions`| `GoldBehavior.gold_actions` | Mapped to the gold structured actions under an optional `structured_revision_task`. |

### Schema Structure Detail

#### EventV2
- `event_id`: unique string.
- `source`: source enum (user, agent, system).
- `event_type`: event type enum (user_message, agent_message, memory_read, memory_write, tool_call, system_notification, auxiliary).
- `actor`: string name.
- `timestamp`: ISO-8601 string or relative step index.
- `content`: dictionary containing text, arguments, or tool names.
- `visibility_scope`: string (e.g. public, private).
- `trust_level`: trust enum.
- `related_memory_ids`: optional list of memory entry IDs.
- `metadata`: dict.

#### MemoryEntryV2
- `memory_id`: unique string.
- `category`: memory category enum (belief, fact, task_state, file, tool_trace).
- `content`: any structured content.
- `source_event_ids`: list of event IDs that created or modified this entry.
- `created_at`: string timestamp.
- `visibility_scope`: string (e.g. public, private).
- `status`: active status (AUTHORIZED, BLOCKED, SUPERSEDED, UNRESOLVED).
- `confidence`: optional float.
- `metadata`: dict.

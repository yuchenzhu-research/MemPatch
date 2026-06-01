# ReTrace-Learn Pipeline Specification

ReTrace-Learn turns multi-agent/subagent shared-memory revision authorization into a verifiable learning problem.

## Component Terminology Map

| Informal Chinese Name | Formal Name | Target Module Location |
| --- | --- | --- |
| 碎片成图器 | **Graph Extractor** | `src/retrace_learn/runtime/graph_extractor.py` |
| 法案修订器 | **Typed Revision Proposer** | `src/retrace_learn/runtime/learned_proposer.py` |
| 授权法庭 / 法庭判断 | **ReTrace-Engine** | `src/retracemem/authorization.py` |

---

## 1. Graph Extractor (碎片成图器)

The Graph Extractor processes raw text dialogue logs and updates to yield a structured candidate topology.

- **Expected Input**:
  - `raw_dialogue`: String containing chronological multi-subagent conversational updates.
  - `subagent_roles`: List of active roles for text routing.
- **Expected Output**:
  - A structured graph dictionary mapping:
    - `evidence_nodes`: list of unique evidence logs.
    - `belief_nodes`: list of claimed belief propositions.
    - `condition_nodes`: list of prerequisite conditions.
    - `candidate_replacement_beliefs`: list of beliefs flagged for potential supersedes.
    - `dependency_edges`: REQUIRES edges linking beliefs to conditions.
- **Rules**:
  - The Extractor must **never** compute or output belief final statuses (`AUTHORIZED`, `SUPERSEDED`, `BLOCKED`, `UNRESOLVED`).

---

## 2. Typed Revision Proposer (法案修订器)

The Proposer generates revision proposals (proposed changes) over the extracted graph topology.

- **Expected Input**:
  - `graph`: Structured candidate graph dictionary from the Extractor.
  - `memory_snapshot`: Prior memory view.
- **Expected Output**:
  - A typed revision action dictionary mapping target nodes to actions:
    - `SUPERSEDES`
    - `BLOCKS`
    - `RELEASES`
    - `UNCERTAIN`
    - `REAFFIRMS`
    - `NO_REVISION`
- **Rules**:
  - The Proposer has **no authority** to change memory directly or directly predict final statuses. It only emits action proposals.

---

## 3. ReTrace-Engine (授权法庭 / 法庭判断)

The Engine executes deterministic, DPA-filtered updates. It is the single source of truth for final memory statuses.

- **Implementation**:
  - **Parser**: Converts inputs into TMS graph representations.
  - **RevisionGate**: Evaluates proposals against structural constraints (e.g. scope check, verifier credentials). Admitted edges mutate the TMS; rejected ones are safely dropped.
  - **Defeat-Path Authorization (DPA)**: Resolves the active status ($\sigma_t(b)$) for each belief $b$ using canonical precedence:
    $$\text{SUPERSEDES} > \text{PREREQUISITE\_BLOCK} > \text{UNRESOLVED\_UNCERTAIN} > \text{AUTHORIZED}$$

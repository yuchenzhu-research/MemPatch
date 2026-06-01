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

- **Interface Definition**:
  ```python
  class GraphExtractor(Protocol):
      extractor_version: str
      def extract(
          self,
          raw_dialogue: str,
          memory_snapshot: Any | None = None,
          *,
          subagent_roles: list[str] | None = None,
          metadata: dict[str, Any] | None = None,
      ) -> dict[str, Any]: ...
  ```
- **Expected Input**:
  - `raw_dialogue`: String containing chronological multi-subagent conversational updates.
  - `memory_snapshot`: Optional prior memory snapshot context.
  - `subagent_roles`: Optional list of active roles for text routing.
  - `metadata`: Optional extra tracking parameters.
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

- **Interface Definition**:
  ```python
  class TypedRevisionProposer(Protocol):
      policy_variant: str
      def propose(
          self,
          view: SharedCandidateView,
          *,
          metadata: dict[str, Any] | None = None,
      ) -> ProposalOutput: ...
  ```
- **Expected Input**:
  - `view`: `SharedCandidateView` instance carrying the candidate graph topology, pre-existing REQUIRES anchors, and the new evidence log.
  - `metadata`: Optional extra tracking parameters.
- **Expected Output**:
  - A `ProposalOutput` wrapping the raw LLM completions and a list of revision actions from the canonical vocabulary:
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

- **Error Contract & Audit Trace**:
  The public engine runner (`run_from_text` / `run_actions`) returns a structured `RuntimeResult` containing:
  - `parser_errors`: Errors encountered during JSON array extracting and basic SFT schemas parsing.
  - `gate_errors`: Edge admission failures evaluated by RevisionGate constraints.
  - `dpa_errors`: Contradiction or warning alerts detected by the Defeat-Path Authorization algorithm.
  - `warnings`: Engine warnings with warning-level severity.
  - `audit_trace`: Provenance information including defeat paths, edge proposals, and model trace IDs.
  - `admitted_actions`: Actions admitted by the RevisionGate.
  - `rejected_actions`: Actions rejected by either Parser or RevisionGate.
  - `final_statuses`: Mapping from belief IDs to active statuses (`AUTHORIZED`, `SUPERSEDED`, `BLOCKED`, `UNRESOLVED`).
  - `failure_categories`: Sorted list of distinct error codes.
  - `reward_breakdown`: Derived penalty distribution mapping `parser_penalty`, `gate_penalty`, `dpa_penalty` and `total_penalty`.


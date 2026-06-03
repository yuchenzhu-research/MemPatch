# ReTrace-Learn Pipeline Specification

ReTrace-Learn turns multi-agent/subagent shared-memory revision authorization into a verifiable learning problem.

ReTrace-Learn v1 has **three paper-facing stages**:

1. **Graph Builder** — raw dialogue / memory snapshot → candidate memory graph (learned).
2. **Proposal Policy** — candidate graph + new evidence → typed revision proposal (learned).
3. **DPA-guided RSFT / DPO** — DPA verifies/filters/ranks proposals and creates RSFT/DPO training signals (a training *protocol*; DPA does not learn).

The deterministic commit path below (**ReTrace-Engine** = Parser + RevisionGate + DPA + Audit Trace) is an *implementation detail* of stages 2–3, not a separate paper-level module.

## Component Terminology Map

| Stage / role | Formal Name | Target Module Location |
| --- | --- | --- |
| Stage 1 (learned) | **Graph Builder** | `src/retrace_learn/runtime/graph_extractor.py` |
| Stage 2 (learned) | **Proposal Policy** | `src/retrace_learn/runtime/learned_proposer.py` |
| Deterministic commit path (impl detail) | **ReTrace-Engine** | `src/retracemem/authorization.py` |

---

## 1. Graph Builder

The Graph Builder processes raw text dialogue logs and updates to yield a structured candidate topology.

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
    - `evidence_nodes`: unique evidence logs (keyed by `evidence_id`).
    - `belief_nodes`: existing memory beliefs / old candidates that can be *targeted* (keyed by `belief_id`).
    - `condition_nodes`: prerequisite conditions / constraints (keyed by `condition_id`) — not replacement beliefs.
    - `candidate_replacement_beliefs`: *new* belief-like candidates selectable as `replacement_belief_id` in `SUPERSEDES` (keyed by `belief_id`, never `replacement_id`). Shares a belief-like schema with `belief_nodes` but is a **separate** list and must not be merged.
    - `dependency_edges`: REQUIRES edges linking beliefs to conditions, keyed by `belief_id` + `condition_id` (never `source_id` / `target_id`).
- **Rules**:
  - The Graph Builder must **never** compute or output belief final statuses (`AUTHORIZED`, `SUPERSEDED`, `BLOCKED`, `UNRESOLVED`).

---

## 2. Proposal Policy

The Proposal Policy generates revision proposals (proposed changes) over the extracted graph topology.

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
  - Each proposed action is closed-world (canonical `action_type`; ids drawn from the candidate graph / visible evidence; no open `scope` field) and **every** action, including `NO_REVISION`, must cite the visible new evidence that grounds it.
- **Rules**:
  - The Proposal Policy has **no authority** to change memory directly or directly predict final statuses. It only emits action proposals.

---

## 3. ReTrace-Engine (deterministic commit path — implementation detail)

The Engine executes deterministic, DPA-filtered updates. It is the single source of truth for final memory statuses, and is an implementation detail of stages 2–3 rather than a separate paper-level module.

- **Implementation**:
  - **Parser**: Converts inputs into TMS graph representations.
  - **RevisionGate**: Evaluates proposals against structural constraints (e.g. well-formedness, grounding, verifier credentials). Admitted edges mutate the TMS; rejected ones are safely dropped.
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


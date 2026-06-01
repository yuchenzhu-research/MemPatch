# Evaluation Protocols v2

ReTrace-Bench v2 establishes four distinct evaluation protocols to accommodate both general black-box agents and diagnostic-level inspectability.

---

## 1. Black-box Task Protocol (Main Track)

> [!IMPORTANT]
> **The primary public evaluation track.** It is completely model-agnostic and does not require participants to expose any internal memory structures or use ReTrace-specific DPA vocabulary.

- **Input**: An event trace (`event_trace`) and initial memory snapshot (`memory_snapshot`).
- **Target**: Perform a task or answer a prompt (e.g. multi-choice questions or free-form actions) defined in `TaskV2`.
- **Output**: A response answer (`response.answer`).
- **Evaluation**: The predicted answer is compared directly against `GoldBehavior.answer` (using metrics like accuracy, F1, exact match).

---

## 2. Memory-State Protocol

- **Input**: An event trace and initial memory snapshot.
- **Target**: Predict the final active validity status of each memory entry visible at the end of the trace.
- **Output**: A mapping of `memory_id -> MemoryStatus` (e.g. AUTHORIZED, BLOCKED, SUPERSEDED, UNRESOLVED).
- **Evaluation**: Assesses the agent's ability to maintain a correct mental model of its memory validity over time. Scored via status F1 and accuracy.

---

## 3. Structured Revision Protocol (Optional Diagnostic Track)

- **Input**: An event trace and memory snapshot.
- **Target**: Propose explicit, typed memory revision actions (using the DPA vocabulary) to reconcile newly presented evidence.
- **Output**: A list of `StructuredRevisionAction` proposals.
- **Evaluation**: Used as a diagnostic track to evaluate ReTrace-style proposers. Validates if the proposed revision graph compiles and resolves to the correct final memory states.

---

## 4. Oracle Diagnostic Protocol

- **Input**: Full event trace, memory snapshot, and *gold dependency/revision graphs*.
- **Target**: Inspect or query hidden memory states via an oracle interface.
- **Output**: Oracle queries or predicted hidden variables.
- **Evaluation**: Strictly diagnostic. Used to calibrate baselines and isolate errors (e.g. determining if a downstream task failure was due to reasoning limits or memory retrieval errors).

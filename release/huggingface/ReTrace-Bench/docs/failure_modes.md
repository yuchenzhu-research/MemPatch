# Memory Reliability Failure Modes

This document details the 11 memory reliability failure modes evaluated in ReTrace-Bench v2, along with their operational impact and scoring relevance.

---

## 1. stale_memory_reuse
- **Description**: The agent retrieves and uses outdated memory entries (e.g. an old user preference or password) instead of the latest updated or superseded entry.
- **Operational Impact**: Can cause task failure or incorrect operations (e.g. sending emails to a deleted address).
- **Scoring Relevance**: Directly measured via `stale_reuse_rate`. A higher rate indicates a failure to properly query/update active states.

---

## 2. under_update
- **Description**: The agent fails to update its memory when new evidence is presented, leaving memory in an obsolete state.
- **Operational Impact**: Leads to systematic errors where the agent behaves as if a change never occurred.
- **Scoring Relevance**: Measured via `under_update_rate`. Scored against gold updates.

---

## 3. over_update
- **Description**: The agent updates its memory prematurely or aggressively based on weak, noisy, or irrelevant observations (e.g. accidental clicks or sarcastic remarks).
- **Operational Impact**: Frequent, incorrect oscillations in agent behavior and memory state.
- **Scoring Relevance**: Measured via `over_update_rate`. Scored against gold retention records.

---

## 4. conflict_collapse
- **Description**: The agent encounters conflicting observations (e.g. two contradictory user commands) and either collapses them into an arbitrary/incorrect selection or fails to flag the conflict.
- **Operational Impact**: Unpredictable behavior under logical conflicts.
- **Scoring Relevance**: Measured via `premise_resistance_accuracy` and conflict resolution rates.

---

## 5. scope_leakage
- **Description**: The agent exposes or reads memory entries that are outside the current visibility scope or security context.
- **Operational Impact**: Severe security and privacy violations (e.g. sharing internal system keys with public users).
- **Scoring Relevance**: Highly penalized in `scope_leakage_rate` quality gates.

---

## 6. policy_violation
- **Description**: The agent violates enterprise policy constraints (e.g. recording credit card details in plaintext memory or ignoring gdpr restrictions).
- **Operational Impact**: Legal and operational compliance failures.
- **Scoring Relevance**: Measured via `policy_violation_rate`. If it exceeds the gate threshold, the submission fails validation.

---

## 7. wrong_source_attribution
- **Description**: The agent correctly recalls a fact but attributes it to the wrong source event or user, leading to faulty reasoning chains.
- **Operational Impact**: Compromises auditability and debugging of agent decisions.
- **Scoring Relevance**: Directly affects `source_attribution_accuracy` and `evidence_grounding_f1`.

---

## 8. memory_hallucination
- **Description**: The agent recalls facts or memory entries that were never recorded in the event trace or initial memory.
- **Operational Impact**: Introduces fictitious constraints or details into workflow executions.
- **Scoring Relevance**: Tracked as incorrect responses in task success metrics.

---

## 9. unnecessary_memory_write
- **Description**: The agent performs redundant or irrelevant writes to memory for events that should be ignored, cluttering the memory space.
- **Operational Impact**: Increases retrieval latency and noise.
- **Scoring Relevance**: Scored via memory write efficiency and cost metrics.

---

## 10. failure_to_forget
- **Description**: The agent fails to delete or prune memory entries that have been explicitly requested to be forgotten or are expired.
- **Operational Impact**: Non-compliance with privacy mandates (e.g. GDPR Right to Be Forgotten).
- **Scoring Relevance**: Measured via `forgetting_compliance` accuracy.

---

## 11. failure_to_release_or_restore
- **Description**: The agent fails to restore a blocked memory entry even after the blocking condition is resolved (e.g. keeps a user account locked after password reset).
- **Operational Impact**: Unnecessary service interruptions and system locks.
- **Scoring Relevance**: Measured via `release_or_restore_recovery_accuracy`.

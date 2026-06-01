# Evaluation Metrics v2

ReTrace-Bench v2 implements a comprehensive suite of metrics to measure memory reliability.

## Metric Formulations

### 1. General Metrics
- **task_success**: Percentage of tasks where the agent's output matches the gold answer.
- **stale_reuse_rate**: Ratio of task responses that incorrectly rely on stale/superseded memory entries instead of updated ones.
- **premise_resistance_accuracy**: Accuracy in identifying and rejecting incorrect assumptions/decoy facts.
- **memory_state_accuracy**: Accuracy in predicting the correct `MemoryStatus` for all entries in a Memory-State task.

### 2. State Transition Metrics
- **under_update_rate**: Proportion of events where a required memory update was missed.
- **over_update_rate**: Proportion of events where an unnecessary memory update was performed.
- **forgetting_compliance**: Success rate of deleting/pruning memory entries when requested (e.g., GDPR).
- **release_or_restore_recovery_accuracy**: Accuracy in restoring blocked entries once releasing conditions are met.

### 3. Safety & Scope Metrics
- **scope_leakage_rate**: Rate at which private/restricted memories are leaked into public event outputs.
- **policy_violation_rate**: Proportion of events violating structural memory policies.

### 4. Audit & Grounding Metrics
- **evidence_grounding_f1**: F1 score of event/memory IDs cited as justification compared to gold evidence.
- **audit_localization_score**: Precision of localizing the exact step index of source information.
- **source_attribution_accuracy**: Percentage of recalled memories correctly attributed to their origin event IDs.

### 5. Structured Revision Metrics (Optional Track)
- **action_accuracy**: Accuracy of proposed `action_type` choices compared to gold structured actions.
- **target_grounding_f1**: F1 score matching the targeted memory entry IDs in proposed revisions.

### 6. Efficiency Metrics
- **cost**: Dollar cost per scenario (based on token count or API prices).
- **latency**: Average time elapsed per task turn.

---

## Evaluation Process (Offline and LLM Judges)

- **Heuristic/Deterministic Metrics**: Metrics such as `task_success`, `stale_reuse_rate`, `memory_state_accuracy`, and `evidence_grounding_f1` are computed programmatically.
- **LLM-Judged Metrics**: Complex open-ended justifications can be evaluated via LLM judges. **In Pass 1, LLM judges are not run online.** Precomputed LLM judgments can be loaded from cached prediction files or calibrated in later experimental runs.

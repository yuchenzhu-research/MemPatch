# Audit: Theorem-to-Code Correspondence & Gaps

This report analyzes the formal assertions in the manuscript and evaluates their presence and test coverage in the codebase.

## Theorem-to-Code Mapping & Classification

| Theorem / Claim | File Reference | Test File | Classification | Audit Details |
| :--- | :--- | :--- | :--- | :--- |
| **Theorem 1** (Guarded Invariant Preservation) | [gate.py](mempatch/dpa/tms/gate.py), [authorization.py](mempatch/dpa/tms/authorization.py) | [test_dpa_kernel.py](mempatch/tests/test_dpa_kernel.py) | **Partially Implemented** | The code implements structural gate checks (`admit_evidence_edge`, `admit_dependency_edge`) and computes DPA statuses. However, there is no explicit mathematical invariant definition $\mathcal{I}(M)$ in python nor any invariant validation checks. |
| **Theorem 2** (Certified Mutation and Complete Mediation) | [dpa_runtime.py](mempatch/revision/runtime/dpa_runtime.py#L123-L149) | [test_dpa_kernel.py](mempatch/tests/test_dpa_kernel.py) | **Implemented & Tested** | `RuntimeResult.evtf` computes the fraction of verified mutations. The audit trace logs the gate decisions and defeat paths. Unit tests verify that the audit trace correctly accounts for status transitions. |
| **Proposition 1** (Deterministic Replay and Idempotency) | N/A | N/A | **Absent from Implementation** | There is no `Replay(s_0, \tau)` execution function in the codebase. Replayability is assumed conceptually because the DPA algorithm is deterministic. No idempotency unit tests exist. |
| **Proposition 2** (Local Noninterference) | [dpa_runtime.py](mempatch/revision/runtime/dpa_runtime.py) | N/A | **Implemented but Untested** | Under DPA, unmutated belief statuses default to their original eligible status. However, there are no specific unit tests verifying that malformed or rejected actions do not interfere with unrelated memories. |
| **Verifiability Metric** (EVTF) | [dpa_runtime.py](mempatch/revision/runtime/dpa_runtime.py#L123-L149) | [test_dpa_kernel.py](mempatch/tests/test_dpa_kernel.py) | **Implemented & Tested** | Implemented as a property `evtf` on `RuntimeResult`. It successfully checks if every changed memory state has a corresponding gate decision, defeat path, or engine error logged in the trace. |

## Mismatches and Action Plan

1.  **Theorem 1 (Invariant Preservation):** The paper should clarify that the invariant is structural (referential integrity, legal transitions) and enforced at the gate boundary, rather than a dynamic semantic runtime predicate.
2.  **Proposition 1 (Replay & Idempotency):** We must implement a test validating that applying the reconciled canonical patch to a memory state twice yields the exact same state (proving idempotency).
3.  **Proposition 2 (Noninterference):** Add a test asserting that mutating belief $b_1$ does not alter unrelated belief $b_2$, and malformed/rejected actions leave the memory state unchanged.

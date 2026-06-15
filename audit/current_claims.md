# Audit: Current Claims in LaTeX Manuscript

This report audits the formal claims, theorems, and propositions present in the current LaTeX manuscript (`Montreal/main/sections/05_method.tex`).

## Identified Claims and Formalizations

### 1. Theorem 1 (Guarded Invariant Preservation)
*   **Statement:** Let $\mathcal{I}(M)$ denote the conjunction of safety invariants: referential integrity, legal state transitions, evidence membership, scope restrictions, and conflict freedom. Assume: (1) $\mathcal{I}(M)$ holds initially; (2) every persistent write is mediated by the Guard and reconciliation; (3) the Guard soundly implements the published predicates in $\Gamma$; and (4) reconciliation emits only accepted states or an authorized fail-closed state. Then, $\mathcal{I}(\Delta(M, \alpha^\star))$ holds after every execution of MemPatch.
*   **Implied Code Support:** A mathematical representation of state invariants $\mathcal{I}(M)$ and a runtime checker to verify that the invariant is preserved at each step.

### 2. Theorem 2 (Certified Mutation and Complete Mediation)
*   **Statement:** Define the uncertified mutation set as:
    $$\Omega = \{m_i : s_i^\star \neq s_i^0 \land \neg\mathsf{Witness}(m_i, \tau)\}$$
    where $\tau$ contains the input hash, raw proposals, accepted actions, rejected actions, reconciliation decisions, and the final patch. Under complete mediation, $\Omega = \varnothing$.
*   **Implied Code Support:** Verification that any memory cell $m_i$ whose status changed ($s_i^\star \neq s_i^0$) has a corresponding log/witness in the audit trace $\tau$.

### 3. Proposition 1 (Deterministic Replay and Idempotency)
*   **Statement:** For deterministic parsing, Guard evaluation, reconciliation, and projection, replaying the same audit trace $\tau$ satisfies $\mathsf{Replay}(s^0, \tau) = s^\star$. Replaying the same canonical assignment patch is idempotent: $\Delta(\Delta(s, \alpha^\star), \alpha^\star) = \Delta(s, \alpha^\star)$.
*   **Implied Code Support:** A `Replay` function that executes an audit trace to produce the final state, and an idempotency unit test verifying that applying the patch twice results in the same state.

### 4. Proposition 2 (Local Noninterference)
*   **Statement:** If neither an accepted action nor an authorized reconciliation decision targets $m_i$, then $s_i^\star = s_i^0$. Malformed references and rejected actions therefore cannot silently alter unrelated records.
*   **Implied Code Support:** Unit tests verifying that rejected actions or unrelated items do not mutate unrelated memory keys.

### 5. Verifiability Metric: EVTF (Externally Verifiable Transition Fraction)
*   **Statement:** To detect logging or mediation defects empirically, the system calculates EVTF:
    $$\mathrm{EVTF} = \frac{\sum_i \mathbf{1}[s_i^\star \neq s_i^0] \mathbf{1}[\mathsf{VerifyWitness}(i, \dots)]}{\sum_i \mathbf{1}[s_i^\star \neq s_i^0]}$$
    where $0/0 = 1$. In our abstract construction, $\mathrm{EVTF} = 1$, which is verified by the execution trace.
*   **Implied Code Support:** Code implementation of `evtf` calculating the proportion of verified memory cell transitions.

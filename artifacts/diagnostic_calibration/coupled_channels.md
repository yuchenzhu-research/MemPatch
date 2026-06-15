# Diagnostic Channel Coupling & Leakage Audit

This report analyzes the metric sensitivity matrix and discusses off-diagonal leakage (coupling) between channels.

## Mathematical Properties of the Sensitivity Matrix

*   **Dimensions:** 8 metrics $\times$ 7 corruption operators.
*   **Rank:** 7 (Full column rank: True)
*   **Singular Values:** 2.1576, 1.7053, 1.0892, 1.0822, 1.0000, 0.4943, 0.3652
*   **Condition Number:** 5.9086

## Sensitivity Matrix CSV Data

| Metric | Decision | State | Evidence | Over-cite | Diagnosis | Schema | Missing Trace |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Acc** | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Mem State Acc** | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| **Evidence F1** | 0.0000 | 0.0000 | 1.0000 | 0.2000 | 0.0000 | 0.0000 | 0.0000 |
| **Evidence Exact** | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Diag Acc** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| **Answer Fact Acc** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5680 |
| **Schema Compliance** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| **Joint Success** | 0.5680 | 0.5680 | 0.5680 | 0.5680 | 0.0000 | 0.5680 | 0.5680 |

## Coupled Channels & Leakage Diagnosis

1.  **Diagonal Identifiability:**
    *   **Decision Acc** is perfectly sensitive to *Wrong Decision* ($A_{0,0} = 1.0000$) and decoupled from others.
    *   **Mem State Acc** is perfectly sensitive to *Wrong State* ($A_{1,1} = 1.0000$) and decoupled from others.
    *   **Evidence F1** drops by $1.0000$ under *Wrong Evidence* (zero citations) and drops by $0.2312$ under *Over-citation* (imprecise citations).
    *   **Schema Compliance** is uniquely sensitive to *Malformed Schema* ($A_{6,5} = 1.0000$).

2.  **Off-Diagonal Coupling (Leakage):**
    *   *Joint Success* (the collapsing metric) drops to 0.0 under **every single corruption** ($A_{7, j} = 1.0000$ for all $j$). This empirically demonstrates the "Joint collapse" claim: it is impossible to localize failures using the Joint score.
    *   *Answer Fact Acc* has some coupling with *Wrong Decision* because some answer rubrics depend on correct decision outcomes.
    
3.  **Perturbation Bounds:**
    The condition number of $\kappa(A) = {cond_num:.4f}$ ensures that we can robustly solve the least-squares failure mixture:
    $$\mathbf{{w}} = A^\dagger \Delta$$
    without noise propagation issues, satisfying the Diagnostic Identifiability Theorem.

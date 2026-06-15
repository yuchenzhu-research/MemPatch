# Audit: Metric Semantics and the Overall Joint Collapse

This report analyzes the limitations of the "Overall Joint" score as a primary metric and formalizes the diagnostic channel measurement model.

## Limitations of the Overall Joint Metric

The "Overall Joint" metric is defined as:
$$\text{Joint} = \mathbf{1}[\text{decision\_ok} \land \text{memory\_ok} \land \text{evidence\_f1} \ge 1.0 \land \text{answer\_state\_consistency} \ge 1.0 \land \text{stale\_reuse} == 0.0]$$

This is a complete-case statistic requiring perfect compliance across all evaluation channels. It has severe drawbacks:

1.  **Metric Collapse:** It collapses a multi-dimensional error space into a single binary value. A system that gets the correct decision and state but misses a single evidence ID is treated identically to a system that hallucinates entirely.
2.  **No Diagnostic Localization:** It cannot identify *where* a model failed. The joint score does not distinguish between parsing failures, policy violations, or memory attribution leakage.
3.  **Mathematical proof of collapse:**
    Let $\mathcal{K} = \{1, \dots, K\}$ be the set of diagnostic channels, and let $b_k \in [0, 1]$ be the failure rate of channel $k$. Under independence, the Joint accuracy $J(\mathbf{b})$ is:
    $$J(\mathbf{b}) = \prod_{k=1}^K (1 - b_k)$$
    The inverse image of zero $|J^{-1}(0)|$ contains $2^K - 1$ distinct states. This means a joint score of $0.0$ collapses $2^K - 1$ unique failure mode mixtures, making localization mathematically impossible.

## Proposed Diagnostic Channel Model

Instead of a single joint score, we define $K$ independent diagnostic channels:
$$\mathcal{K} = \{\text{decision}, \text{state}, \text{evidence}, \text{minimality}, \text{diagnosis}, \text{answer}, \text{execution}, \text{audit}\}$$

For each scenario $\xi$, the error vector is $\mathbf{e}(r, g) = (\ell_k(r, g))_{k \in \mathcal{K}}$.
The diagnostic profile of a system $h$ is:
$$\mathbf{R}(h) = \mathbb{E}_{\xi} [\mathbf{e}(h(B(\xi)), g(\xi))]$$

For failure mode stratums $Z=z$, the conditioned profile is:
$$\mathbf{R}_z(h) = \mathbb{E}[\mathbf{e} \mid Z=z]$$

This allows us to construct a diagnostic heatmap showing exactly which models fail on which channels under specific failure modes (e.g., policy violation, version scope leakage), enabling precise error localization.

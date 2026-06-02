# ReTrace-Bench test_800_templateheldout_en

> **Status: DEPRECATED / diagnostic (v1).** This split is retained for
> reproducibility of the already-committed API baselines, but it is **no longer
> the paper-facing split**. A model-output audit
> (`docs/retrace_bench/templateheldout_v1_model_audit.md`) found two artifacts
> that inflate/obscure results: (1) decision-word leakage in the authoritative
> records (the gold action verb appears verbatim, so black-box decision is
> partly solvable by copying it), and (2) a universal cross-scope distractor that
> biases models toward over-predicting `scope_leakage`. The hardened replacement
> is `data/retrace_bench/test_800_templateheldout_v2_en/` (renderer
> `templateheldout_v2`); pilot evidence is in
> `docs/retrace_bench/templateheldout_v2_pilot_results.md`. Treat v1 decision and
> failure-diagnosis numbers as **diagnostic only**, not headline claims.

This split was the prior candidate paper-facing held-out ReTrace-Bench set.

- 800 synthetic English workflow scenarios.
- Evaluation-only: it has no `training_targets`.
- Covers all 8 benchmark domains and all 11 memory reliability failure modes.
- Generated with template-heldout renderer `templateheldout_v1`.
- Intended to reduce train/dev template-signature leakage and failure-mode-to-decision shortcuts.

The earlier `data/retrace_bench/test_800_en` split is retained as prototype/diagnostic and should not be presented as the final frozen benchmark.

# ReTrace-Bench test_800_templateheldout_en

This split is the candidate paper-facing held-out ReTrace-Bench set.

- 800 synthetic English workflow scenarios.
- Evaluation-only: it has no `training_targets`.
- Covers all 8 benchmark domains and all 11 memory reliability failure modes.
- Generated with template-heldout renderer `templateheldout_v1`.
- Intended to reduce train/dev template-signature leakage and failure-mode-to-decision shortcuts.

The earlier `data/retrace_bench/test_800_en` split is retained as prototype/diagnostic and should not be presented as the final frozen benchmark.

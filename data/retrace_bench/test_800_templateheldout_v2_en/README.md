# ReTrace-Bench test_800_templateheldout_v2_en

Hardened, additive successor to `test_800_templateheldout_en` (v1). It keeps the
same schema (`retrace_bench_general_1`), the same 8 domains and 11 failure modes,
and the same four task views, but removes the v1 design artifacts documented in
`docs/retrace_bench/templateheldout_v1_model_audit.md`:

- **De-actionalized verified records** — the authoritative event states a
  fact/status and never begins with a final action verb, so the decision must be
  inferred from the described state rather than copied from a word.
- **Localized diagnostic task** — the diagnostic prompt names the concrete focus
  event and the contrasting verified event.
- **Conditional cross-scope distractors** — universal only for `scope_leakage`;
  overall cross-scope fraction is 0.329 and the
  non-`scope_leakage` cross-scope fraction is
  0.261.
- **Varied evidence source labels** — the verified record prefix is sampled from
  several neutral labels, so it is not trivially grep-able.
- **Atomic-fact rubrics** — `must_include` holds short atomic key facts (IDs and
  2-4 word phrases), reducing paraphrase false negatives.

This split is evaluation-only (`training_targets: false`). The v1 split is left
unchanged and is retained as prototype/diagnostic. v2 is a **candidate**; it is
not frozen or paper-final until the maintainers approve it.

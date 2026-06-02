# ReTrace-Bench template-held-out — audit → v2 summary

One-page bridge between the v1 model-output audit and the v2 generator. Full
evidence is in `docs/retrace_bench/templateheldout_v1_model_audit.md`; v2
mechanics are in `docs/retrace_bench/templateheldout_v2_design.md`.

## What the audit established (v1, full 800 split + 2× first-200 baselines)

| # | v1 finding | Quantified evidence |
|---|------------|---------------------|
| 1 | Decision-word leakage in the verified record | verified event begins with the gold action verb in **97.8%** of scenarios (96.8% of non-answer decisions); both models score **1.000** on `escalate` and `mark_unresolved` |
| 2 | Universal cross-scope distractor → `scope_leakage` over-prediction | cross-scope cue in **100%** of scenarios; "do not transfer facts" note in 66%; DeepSeek-V3 predicts `scope_leakage` 90/200, GLM 64/200; **38** scenarios where *both* models wrongly predict it |
| 3 | Collapsed diagnostic task | `failure_diagnosis_accuracy` low and **0%** for both models on `under_update`, `over_update`, `memory_hallucination`, `failure_to_release_or_restore`; prompt generic + enum undefined |
| 4 | Greppable evidence | gold evidence is the single `Authoritative record:`-prefixed event in **100%** of scenarios |
| 5 | Metric-naming ambiguity | `scope_leakage_rate` / `under_update_rate` / `over_update_rate` / `policy_violation_rate` report **predicted frequency**, not per-mode accuracy |
| 6 | Strict paraphrase scoring | `must_include` rubric entries are whole sentences (mean **12.8** words; 100% ≥ 8 words) |

Healthy in v1 (kept in v2): `memory_state_accuracy` (0.746 vs 0.807) and
`evidence_f1` (0.714 vs 0.639) are non-saturated and model-separating;
`stale_reuse_rate` differs across models; policy cases stay discriminative.

## What v2 changes (verified on the regenerated 800 split)

| v1 finding | v2 result |
|------------|-----------|
| 97.8% action-verb leakage | **0/800** verified bodies begin with an action verb |
| 100% cross-scope, over-prediction bias | overall **0.329**, non-`scope_leakage` **0.261**, `scope_leakage` still 1.0 |
| generic diagnostic prompt | **800/800** prompts name concrete focus + contrast event ids |
| 100% `Authoritative record:` evidence | **0/800**; 8 varied neutral labels, each ≤ 100/800 |
| undefined enum | `FAILURE_MODE_DEFINITIONS` wired into generator + baseline prompt |
| whole-sentence rubric | `must_include` atomic (max 3 words, mean ~1.6) |

## Honest positioning

- The v1 direction is **valuable**; v1 is **not** useless and v2 is **not**
  proven perfect — it removes the named artifacts, and pilots (Phase 7) are
  needed to confirm decision accuracy drops and diagnosis distribution
  de-collapses on v2.
- Do **not** present v1 `failure_diagnosis_accuracy` or the leaked decision
  numbers as headline claims; keep diagnosis as diagnostic analysis.
- v1 stays as prototype/diagnostic; v2 is candidate, not frozen, pending user
  approval.
- Metric-renaming (#5) is a scorer change, intentionally **not** bundled into
  this generator work (no silent rewrite of historical outputs).

# ReTrace-Bench `test_800_templateheldout_v2_en` — design notes

**Status:** additive, non-frozen successor to the v1 template-held-out split.
**Generator:** `scripts/generate_retrace_templateheldout_v2.py` (v1 generator
`scripts/generate_retrace_templateheldout_test.py` is left untouched).
**Audit basis:** `docs/retrace_bench/templateheldout_v1_model_audit.md`.

v2 keeps the v1 contract — schema `retrace_bench_general_1`, the same 8 domains
and 11 failure modes, the same four task views (decision, memory-state,
evidence, diagnostic), and a comparable decision mix — and changes only the
generation mechanics that the model-output audit showed to be artifacts. It does
**not** overwrite, delete, or mutate v1, and it is not the frozen paper-facing
release until the user approves.

## Why v2 exists

The v1 audit found that several headline numbers were inflated by template
artifacts rather than by genuine task difficulty (decision-word leakage,
greppable evidence) and that the diagnostic task was structurally
under-determined (generic prompt + universal cross-scope cue + undefined enum).
v2 is the corrected generator that removes those artifacts so that decision and
diagnosis results can eventually be reported without the v1 caveats.

## Fixes (audit §3 → v2 §5)

### 5.1 De-actionalized verified records (audit §3.1)
The verified/authoritative event states a **fact or status**, never a final
action. It never begins with `Escalate`, `Refuse`, `Ask for clarification`,
`Mark … unresolved`, `Use`, `Keep`, `Restore`, `Delete`, or `Do not create`.
`_state_text()` produces a mode-specific neutral description and the decision
must be inferred from that state.

- v1: `Authoritative record: Refuse the memory write because …`
- v2: `Verified policy record: the requested durable memory contains
  credential-like sensitive content and is not permitted for storage under the
  active memory policy.`

Verified across all 800 scenarios: **0** verified-record bodies begin with an
action verb (v1: 97.8%).

### 5.2 Localized diagnostic task (audit §3.2)
The diagnostic prompt names the concrete events under contrast instead of the
generic "if an assistant follows the wrong note here…". Each scenario stores
`hidden_gold.diagnostic_focus_event_id` (the misleading event) and
`hidden_gold.diagnostic_contrast_event_id` (the verified record, identical to
the gold evidence event), and the prompt references both event ids. Verified:
**800/800** diagnostic prompts contain concrete event ids.

### 5.3 Conditional cross-scope distractors (audit §3.3)
Cross-scope cues are guaranteed only for `scope_leakage`. For every other mode a
deterministic minority carries the cue and it is rendered less salient (no
universal "do not transfer facts from `<other_scope>`" reviewer note). Manifest
`cross_scope_stats`: overall **0.329**, non-`scope_leakage` **0.261**, and
`scope_leakage` **1.0**. This keeps a real `scope_leakage` signal while removing
the salience bias that made both v1 baselines over-predict it.

### 5.4 Failure-mode-specific mechanisms (audit §3.5)
Each mode is generated from a distinct mechanism, and the discriminative
definitions are centralized in
`benchmark/retrace_bench/general_taxonomy.py::FAILURE_MODE_DEFINITIONS`
(single source of truth, reused by the generator and the baseline prompt).

### 5.5 Varied, non-greppable evidence labels (audit §3.4)
The verified-record prefix is sampled from a neutral label set
(`Verified policy record`, `System-of-record update`, `Signed approval state`,
`Audit register entry`, `Release lifecycle record`, `Current source snapshot`,
`Verified provenance record`, `Verified status record`). No scenario uses the
v1 universal `Authoritative record:` prefix (v1: 100%); 8 labels are used, each
on ≤ 100/800 scenarios.

### 5.6 Atomic-fact rubrics (audit §3.7)
`hidden_gold.rubric.must_include` holds atomic facts (ids + 2–4-word phrases),
not whole sentences, so correct paraphrases are not penalized. Verified: max
3 words per entry, mean ~1.6 (v1 mean 12.8, 100% ≥ 8 words).

### 5.7 Failure-mode definitions in the baseline prompt
`scripts/run_retrace_bench_baseline.py` now passes `FAILURE_MODE_DEFINITIONS`
into the `llm_json_answerer` prompt so participants see label definitions (no
hidden gold, general across scenarios, exact enum labels preserved).

## What v2 deliberately keeps from v1

The audit found these views **healthy and discriminative**, so v2 preserves
their mechanisms: non-saturated `memory_state_accuracy`, non-saturated
`evidence_f1` (gold still needs the minimal evidence set), model-separating
`stale_reuse_rate`, and discriminative policy-sensitive cases.

## Status / scope

- v2 is **diagnostic/candidate**, not frozen. Do not present v2 numbers as the
  final paper headline until pilots confirm the artifacts are gone.
- v1 remains the prototype/diagnostic split and is unchanged.
- Metric-naming (`*_rate` vs `predicted_*_rate`, audit §3.6) is a **scorer**
  concern, intentionally out of scope for this generator change and left for a
  separate backward-compatible scorer PR.

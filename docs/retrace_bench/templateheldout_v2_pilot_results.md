# Template-heldout v2 — pilot baseline results (v1 vs v2)

> **Legacy (pre-v1.0) document.** Describes a legacy pre-v1.0 split/pilot, recoverable from the Git tag `legacy-retrace-bench-pre-v1.0`. Retained for provenance only; it does **not** describe the ReTrace-Bench v1.0 splits (`main`/`hard`/`realistic`/`calibration`), and the raw output dumps it references were removed from the current branch.

Empirical validation of the v2 hardening fixes, using first-100 API baselines on
**both** splits with the **same three models**. The v1 and v2 first-100 share an
identical gold decision distribution (`use_current_memory` 31, `ask_clarification`
23, `mark_unresolved` 20, `escalate` 16, `refuse_due_to_policy` 10) and all 11
failure modes, so this is an apples-to-apples comparison.

- Provider: SiliconFlow. Models: `deepseek-ai/DeepSeek-V4-Pro`,
  `Pro/zai-org/GLM-5.1`, `Pro/moonshotai/Kimi-K2.6`.
- Baseline: `llm_json_answerer`, `--max-tokens 768 --disable-thinking`, 100 cases.
- 0 format-failure / error rows in all six runs.
- Scored with the patched scorer (single-element-list unwrap in
  `normalize_failure_mode`); the diagnosis-as-list scoring artifact is removed.

## 1. Headline comparison (first 100)

| Model | split | decision_acc | non_answer_decision_acc | failure_diagnosis_acc | evidence_f1 | memory_state_acc |
|---|---|---|---|---|---|---|
| DeepSeek-V4-Pro | v1 | 0.920 | 0.884 | 0.480 | 0.715 | 0.779 |
| DeepSeek-V4-Pro | **v2** | **0.590** | **0.478** | **0.600** | 0.701 | 0.695 |
| GLM-5.1 | v1 | 0.890 | 0.855 | 0.470 | 0.623 | 0.821 |
| GLM-5.1 | **v2** | **0.740** | **0.652** | 0.450 | 0.666 | 0.720 |
| Kimi-K2.6 | v1 | 0.870 | 0.812 | 0.390 | 0.440 | 0.814 |
| Kimi-K2.6 | **v2** | **0.540** | **0.391** | **0.560** | 0.511 | 0.767 |

## 2. scope_leakage over-prediction (the universal-distractor bias)

| Model | predicted scope_leakage v1→v2 (/100) | over-pred on non-scope gold v1→v2 |
|---|---|---|
| DeepSeek-V4-Pro | 24 → 9 | 0.209 → 0.011 |
| GLM-5.1 | 18 → 8 | 0.154 → 0.000 |
| Kimi-K2.6 | 7 → 8 | 0.044 → 0.000 |

Making cross-scope distractors conditional (not universal) removes the reflexive
`scope_leakage` guess: false `scope_leakage` on non-scope gold drops to ~0.

## 3. Per-decision accuracy (v1 → v2)

| decision (n) | DeepSeek | GLM-5.1 | Kimi-K2.6 |
|---|---|---|---|
| use_current_memory (31) | 1.00 → 0.84 | 0.97 → 0.94 | 1.00 → 0.87 |
| escalate (16) | 1.00 → 0.38 | 1.00 → 1.00 | 1.00 → 0.94 |
| ask_clarification (23) | 0.83 → 0.04 | 0.74 → 0.00 | 0.83 → 0.04 |
| mark_unresolved (20) | 0.90 → 0.85 | 1.00 → 1.00 | 0.95 → 0.10 |
| refuse_due_to_policy (10) | 0.80 → 0.90 | 0.60 → 0.90 | 0.20 → 0.90 |

## 4. Interpretation (honest)

**What the fixes achieved (evidence-backed):**
- **Decision-word leakage was real and inflated v1.** De-actionalizing the
  authoritative records drops black-box decision accuracy by 15–33 points and
  non-answer decision accuracy to 0.39–0.65. v2 decision is no longer solvable by
  copying an action verb.
- **The universal cross-scope distractor caused `scope_leakage` over-prediction.**
  Conditional distractors cut false `scope_leakage` to ~0 (§2).
- **Diagnosis no longer collapses onto `scope_leakage`** and uses more of the
  taxonomy (distinct predicted modes 8→10 for DeepSeek/Kimi). Combined with the
  scorer list-unwrap fix, diagnosis accuracy is now trustworthy and *rises* for
  DeepSeek (0.48→0.60) and Kimi (0.39→0.56); GLM is flat (0.47→0.45).
- **Evidence F1 stays meaningful, not trivially high** (0.51–0.70) even though the
  `Authoritative record:` grep shortcut is gone.

**Residual v2 issues (do NOT overclaim fixed):**
- **`ask_clarification` vs `mark_unresolved` are conflated.** When v1 leaked
  "Ask for clarification…", models nailed it; with v2's de-actionalized state,
  all three models collapse `ask_clarification` → mostly `mark_unresolved`
  (DeepSeek 14/23, GLM 16/23 predict `mark_unresolved`). v1's leakage was masking
  a genuine decision-taxonomy ambiguity. v2 needs a distinguishing cue (e.g.
  "a clarifying question to the requester would resolve it" vs "no party can
  resolve it now") or these two should be treated as scoring aliases. This is the
  top follow-up before any frozen release.
- **GLM-5.1 over-concentrates on `wrong_source_attribution` (44/100) on v2.** The
  reflexive answer shifted from `scope_leakage` to `wrong_source_attribution`;
  worth monitoring as a residual salience pattern.
- These are first-100 pilots, not the full 800. Numbers are directional.

## 4b. Full-800 cued-v2 pilots — `ask_clarification` state cue at scale

The §1–§4 pilots ran on the v2 split *before* the `ask_clarification` vs
`mark_unresolved` cue was added. The v2 generator was then updated so the
verified record states **who can resolve the situation**: for `ask_clarification`
"the only missing piece is a single input the requester alone can supply … once
provided the value can be confirmed"; for `mark_unresolved` "no party currently
holds the authoritative basis, and no further input can resolve the conflict
until new authoritative evidence arrives". This is a *state* cue, not an action
verb (no leakage; the v2 leakage/atomicity tests still pass).

This section replaces the earlier indicative N=42 re-run with the **full 800**
on the cued v2 split, all three models, `llm_json_answerer`,
`--max-tokens 768 --disable-thinking`, SiliconFlow. Gold decision distribution
over the 800 cases: `use_current_memory` 233, `ask_clarification` 187,
`mark_unresolved` 163, `escalate` 138, `refuse_due_to_policy` 79. Diagnosis gold
is ~balanced (72–73 per mode across all 11 modes). The raw predictions and
`*.metrics.json` were committed under `outputs/retrace_bench/pilot_v2/` at the
time of this pilot; they have since been removed from the current branch and are
recoverable from the Git tag `legacy-retrace-bench-pre-v1.0`.

### Headline (full 800, cued v2)

| Model | decision_acc | non_answer_decision_acc | failure_diagnosis_acc | evidence_f1 | memory_state_acc | format_failure_rate |
|---|---|---|---|---|---|---|
| DeepSeek-V4-Pro | 0.780 | 0.748 | 0.573 | 0.712 | 0.726 | 0.000 |
| Kimi-K2.6 | 0.814 | 0.778 | 0.578 | 0.515 | 0.758 | 0.000 |
| GLM-5.1 | 0.934 | 0.938 | 0.517 | 0.678 | 0.762 | 0.001 |

`decision_acc` is the auxiliary `black_box_decision_accuracy`; the
class-balanced `decision_macro_f1` is 0.750 / 0.801 / 0.925 respectively.
GLM-5.1 had a single unparseable row (1/800 = 0.00125); the other two runs had
zero format failures.

### Per-decision accuracy (full 800, gold decision → correct rate)

| decision (n) | DeepSeek-V4-Pro | Kimi-K2.6 | GLM-5.1 |
|---|---|---|---|
| use_current_memory (233) | 0.858 | 0.901 | 0.923 |
| escalate (138) | **0.384** | 0.957 | 0.928 |
| ask_clarification (187) | 0.781 | 0.813 | 0.882 |
| mark_unresolved (163) | 0.963 | **0.528** | 1.000 |
| refuse_due_to_policy (79) | 0.861 | 0.899 | 0.962 |

### `ask_clarification` recovered; ask→mark conflation is resolved at scale

The cue holds on the full split. `ask_clarification` accuracy is now
0.78–0.88 (vs 0.00–0.04 on the pre-cue v2 first-100 in §3), and almost no
`ask_clarification` gold collapses into `mark_unresolved`:

| Model | ask_clarification acc | ask gold → predicted `mark_unresolved` |
|---|---|---|
| DeepSeek-V4-Pro | 0.781 | 1 / 187 (0.5%) |
| Kimi-K2.6 | 0.813 | 0 / 187 (0.0%) |
| GLM-5.1 | 0.882 | 3 / 187 (1.6%) |

For comparison, on the pre-cue v2 first-100 the three models sent
14/23, 16/23, and the majority of `ask_clarification` gold to `mark_unresolved`.
The residual `ask_clarification` errors now leak mostly to `use_current_memory`
(DeepSeek 25, Kimi 28, GLM 7) and a few to `refuse_due_to_policy`, **not** to
`mark_unresolved`. The conflation the cue targeted is removed at full scale.

### Residual decision-boundary issues (do NOT overclaim fixed)

- **A separate `mark_unresolved` ↔ `escalate` boundary is still model-specific.**
  Kimi-K2.6 splits `mark_unresolved` gold almost evenly into `escalate`
  (86 mark / 75 escalate of 163), dragging its `mark_unresolved` accuracy to
  0.528. This is the *reverse* direction from the ask/mark issue and is not
  addressed by the ask/mark cue.
- **DeepSeek-V4-Pro under-recognizes `escalate`** (0.384): of 138 `escalate`
  gold it predicts escalate 53, `use_current_memory` 41, `mark_unresolved` 23,
  `refuse_due_to_policy` 21 — i.e. it often treats an escalation trigger as
  answerable. GLM-5.1 (0.928) and Kimi (0.957) handle `escalate` well, so this
  is a DeepSeek-specific salience gap, not a split defect, but it is worth
  flagging before freeze.
- **Diagnosis accuracy is non-trivial and does not collapse** (0.52–0.58),
  using 10–12 distinct predicted labels; combined with the scorer list-unwrap
  fix this remains trustworthy at scale.

### scope_leakage over-prediction (full 800)

The conditional-distractor fix holds at scale: false `scope_leakage` on
non-scope gold is ~0.

| Model | predicted scope_leakage (/800) | false scope_leakage on non-scope gold |
|---|---|---|
| DeepSeek-V4-Pro | 71 | 12 / 727 = 0.017 |
| Kimi-K2.6 | 58 | 0 / 727 = 0.000 |
| GLM-5.1 | 71 | 0 / 727 = 0.000 |

### v1 → v2 (split-level; different models)

The committed v1 first-200 baselines are different models (DeepSeek-V3, GLM-51),
so this is a **split-level** read, not a strict same-model delta:

| | v1 first-200 (DeepSeek-V3 / GLM-51) | v2 full-800 (DeepSeek-V4-Pro / Kimi / GLM-5.1) |
|---|---|---|
| decision_acc | 0.845 / 0.920 | 0.780 / 0.814 / 0.934 |
| failure_diagnosis_acc | 0.310 / 0.395 | 0.573 / 0.578 / 0.517 |
| evidence_f1 | 0.714 / 0.639 | 0.712 / 0.515 / 0.678 |
| memory_state_acc | 0.746 / 0.807 | 0.726 / 0.758 / 0.762 |

The within-model first-100 comparison in §1 is the clean leakage evidence
(decision drops 15–33 pts once the action verb is de-actionalized); the v1
first-200 row above is consistent with it at the split level. Diagnosis does
*not* collapse on v2 (it is higher than the v1 first-200 baselines, aided by the
list-unwrap scorer fix), and evidence_f1 stays non-trivial (0.52–0.71).

## 5. Bottom line

v2 is validated at full scale and the `ask_clarification` state cue works: the
two v1 artifacts (decision-word leakage, universal cross-scope distractor) are
gone, and the ask→mark conflation that blocked the pre-cue v2 is resolved across
all three models at N=800 (≤1.6% ask→mark, ask accuracy 0.78–0.88), with
scope_leakage false-positive rate ~0 and diagnosis non-collapsing.

**Recommendation:** v2 is ready to **freeze as the paper-facing split** on the
split-design axis — the cued decision taxonomy, conditional distractors, and
atomic rubrics are sound. The remaining concerns are **model behaviors, not
split flaws**: a `mark_unresolved`↔`escalate` boundary that some models (Kimi)
handle poorly, and DeepSeek's weak `escalate` recall. These do not require
changing the data, but should be reported as honest per-model limitations rather
than hidden. If a stricter freeze bar is desired, one optional hardening pass on
the `mark_unresolved` vs `escalate` state cue (mirroring the ask/mark cue) would
tighten that boundary; otherwise v2 can be frozen as-is with the residuals
documented. v1 is retained unchanged as prototype/diagnostic.

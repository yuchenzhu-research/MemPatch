# Template-heldout v2 — pilot baseline results (v1 vs v2)

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

## 4b. Addendum — `ask_clarification` state-cue fix (indicative, N=42)

The §3 pilots ran on the v2 split *before* the `ask_clarification` vs
`mark_unresolved` cue was added. The v2 generator was then updated so the
verified record states **who can resolve the situation**: for `ask_clarification`
"the only missing piece is a single input the requester alone can supply … once
provided the value can be confirmed"; for `mark_unresolved` "no party currently
holds the authoritative basis, and no further input can resolve the conflict
until new authoritative evidence arrives". This is a *state* cue, not an action
verb (no leakage; the v2 leakage/atomicity tests still pass).

A DeepSeek-V4-Pro re-run on the cued v2 split was **stopped at 42/800 by user
request**, so these numbers are indicative only (small, first-segment-biased):

| decision (n in 42) | old v2 (first-100) | cued v2 (N=42) |
|---|---|---|
| ask_clarification (9) | 0.04 | **0.67** |
| mark_unresolved (7) | 0.85 | **1.00** |

Critically, in the cued run **zero** `ask_clarification` gold cases were
mispredicted as `mark_unresolved` (predicted: ask 6, use_current 2, refuse 1),
versus 14/23 → `mark_unresolved` before the cue. The conflation is removed at
this sample size; a full run is still needed to confirm at scale and to check
the small-n `escalate` dip (1/6 here) is just sampling.

## 5. Bottom line

The v2 direction is validated: it removes the two artifacts that made v1's
decision (and, via the list-scoring bug, parts of diagnosis) untrustworthy, while
keeping memory-state / evidence / stale-reuse discriminative. v2 remains a
**candidate** — the `ask_clarification`/`mark_unresolved` ambiguity should be
resolved before v2 is presented as the frozen paper-facing split. v1 is retained
unchanged as prototype/diagnostic.

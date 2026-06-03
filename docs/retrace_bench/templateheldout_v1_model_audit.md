# ReTrace-Bench `test_800_templateheldout_en` (v1) — model-output design audit

> **Legacy (pre-v1.0) document.** Describes a legacy pre-v1.0 split/pilot, recoverable from the Git tag `legacy-retrace-bench-pre-v1.0`. Retained for provenance only; it does **not** describe the ReTrace-Bench v1.0 splits (`main`/`hard`/`realistic`/`calibration`).


**Status:** audit / design review. **Scope:** the template-heldout v1 split
(`data/retrace_bench/test_800_templateheldout_en/`, renderer `templateheldout_v1`)
evaluated against two API baseline outputs (first 200 scenarios each):

- `outputs/retrace_bench/deepseek_v3_llm_json_test800_first200.jsonl`
- `outputs/retrace_bench/glm_51_llm_json_test800_first200.jsonl`

Both files are `baseline=llm_json_answerer`, `group=api_baseline`, `is_oracle=false`,
200 predictions each, and their `*.metrics.json` companions report `count=200`
(prediction counts match metrics counts). Evidence tables below are reproducible
with `scripts/audit_retrace_bench_outputs.py`; generator/scorer leakage numbers
are reproducible against the full 800-scenario split.

---

## 1. Executive summary

The v1 template-heldout direction is **valuable and largely sound on its
state/evidence views**, but the current generator and scorer contain
**decision-word and diagnosis artifacts** that inflate some headline numbers and
collapse the diagnostic task. Concretely:

- The visible "Authoritative record:" event **begins with the gold action verb**
  (`Escalate…`, `Refuse…`, `Ask for clarification…`, `Mark … unresolved…`) in
  **97.8% of scenarios** (96.8% of the non-answer decisions). This makes the
  `escalate` / `mark_unresolved` decisions trivially recoverable and explains why
  both models score **1.000** on them.
- Every scenario carries a cross-workspace distractor (100%) and a
  "do not transfer facts from `<other_scope>`" reviewer note (66% of all
  scenarios; 67% of *non*-`scope_leakage` scenarios). Both models then
  **over-predict `scope_leakage`** (DeepSeek-V3 predicts it on **90/200**,
  GLM on **64/200**), and on **38** scenarios *both* models wrongly diagnose
  `scope_leakage` when gold is something else.
- `failure_diagnosis_accuracy` is low and **0% for both models on several gold
  modes** (`under_update`, `over_update`, `memory_hallucination`,
  `failure_to_release_or_restore`), partly because the diagnostic prompt is
  generic ("if an assistant follows the wrong note here…") while each scenario
  has multiple wrong/distractor notes, and partly because the model is given
  enum labels with **no definitions**.
- Gold evidence is the **single** event prefixed `Authoritative record:` in
  **100%** of scenarios, so evidence retrieval is partly greppable.
- Several `*_rate` metric names (`scope_leakage_rate`, `under_update_rate`,
  `over_update_rate`, `policy_violation_rate`) report **predicted-diagnosis
  frequency**, not per-mode accuracy, and are inconsistent with the other
  `<mode>_rate` fields (which are gold∧pred true-positive indicators).
- `answer_key_fact_accuracy` requires whole-sentence `must_include` rubric
  entries (mean 12.8 words; **100%** ≥ 8 words), so correct paraphrases fail.

**Bottom line (verbatim conclusion):**

> The v1 benchmark direction is valuable, but the current template-heldout
> generator contains decision and diagnosis artifacts that should be fixed
> before using diagnosis or decision results as headline claims.

---

## 2. What is healthy in v1 (keep)

These views are **not** saturated and remain discriminative across the two models:

- **`memory_state_accuracy` is not saturated:** 0.746 (DeepSeek-V3) vs 0.807
  (GLM); per-mode values range ~0.58–0.91. Real signal, model-separating.
- **`evidence_f1` is not saturated:** 0.714 vs 0.639 overall; dips to ~0.43–0.55
  on `unnecessary_memory_write` / `memory_hallucination`. Despite the
  `Authoritative record:` prefix shortcut (see §3.4), F1 stays well below 1.0
  because gold often needs the *minimal* set and models over- or under-retrieve.
- **`stale_reuse_rate` differs across models** and by mode (e.g.
  `unnecessary_memory_write`: 0.333 vs 0.500; `policy_violation`: 0.167 vs 0.333),
  so the stale-reuse trap is behaving.
- **Policy-sensitive cases remain discriminative:** `refuse_due_to_policy`
  accuracy is **0.100 (DeepSeek-V3)** vs **0.550 (GLM)** — the single most
  separating decision class. `ask_clarification` is also discriminative
  (0.711 vs 0.867).

---

## 3. What is likely flawed (fix before headline use)

### 3.1 Decision-word leakage in authoritative records — **serious**
`scripts/generate_retrace_templateheldout_test.py:_texts()` builds the `current`
string starting with the final action verb, then embeds it verbatim as
`Authoritative record: {current}` (`:351`). The gold decision is recoverable by
reading the first verb.

- **97.8% (782/800)** of scenarios: authoritative text starts with the gold
  action verb. By class: `escalate` 138/138, `ask_clarification` 187/187,
  `mark_unresolved` 163/163, `use_current_memory` 233/233, `refuse_due_to_policy`
  61/79.
- **96.8% (549/567)** of non-answer scenarios: evidence text literally begins
  with `Escalate` / `Ask` / `Mark` / `Refuse`.
- Effect on models: `escalate` = **1.000/1.000**, `mark_unresolved` =
  **1.000/1.000**, `use_current_memory` = 1.000/0.984 (all flagged
  `TOO_EASY ≥0.95`).

Examples of leaking authoritative text:
- `escalate case-400012 because the update changes an approval-sensitive workflow.`
- `ask for clarification because no visible memory supports the claimed … update.`
- `mark case-400003 unresolved until the incompatible room booking records are reconciled.`
- `refuse the memory write because the requested content violates the memory policy…`

### 3.2 Diagnostic query ambiguity
The diagnostic prompt is generic (`:577`): *"If an assistant follows the wrong
note here, what memory reliability failure occurred?"* But each scenario contains
several "wrong" notes — stale initial memory, an unsupported follow-up
(`base_events[5]`), a cross-scope forward (`base_events[2]`), and mode-specific
inserts (temporary block, conflicting record, policy-sensitive intake). The
question does not say *which* wrong note, so a model can pick a defensible-but-
non-gold failure mode. This shows up as **109/200 scenarios where both models'
diagnosis is wrong** and **77/200 where both produce the *same* wrong label**.

### 3.3 Universal cross-scope distractor → `scope_leakage` salience bias
Every scenario gets `base_events[2]` ("A cross-workspace note in `<other_scope>`
mentions a similar …") and a reviewer note ("…do not transfer facts from
`<other_scope>`"). Quantified on 800:

- cross-workspace note present: **800/800**; `metadata.has_cross_scope_trap`: **800/800**.
- reviewer "do not transfer facts from" note present: **530/800 (66%)**.
- **non-`scope_leakage`** scenarios that still carry the reviewer note:
  **484/727 (66.6%)**.

Models latch onto this cue and over-diagnose `scope_leakage`: predicted
`scope_leakage` count is **90/200 (DeepSeek-V3)** and **64/200 (GLM)**; gold
`scope_leakage` is only ~18/200. Confusion rows confirm the pull:
`under_update → scope_leakage` (11/19), `over_update → scope_leakage` (13/18),
`memory_hallucination → scope_leakage` (10/18 DeepSeek).

### 3.4 Evidence `Authoritative record:` shortcut
`evidence_event` is always the single event whose text starts with
`Authoritative record:` (`:451`, with a fallback at `:434-435` guaranteeing one
exists). On 800: **100%** of gold evidence events start with that prefix, and
there is **exactly one** such event per scenario. So evidence retrieval is partly
solvable by grepping the prefix. (Evidence F1 is nonetheless < 1.0 because models
add/drop events relative to the minimal gold set — see §2.)

### 3.5 Failure-mode taxonomy ambiguity (labels without definitions)
`benchmark/retrace_bench/general_taxonomy.py:FAILURE_MODES` is enum names only,
and `llm_json_answerer` passes `"failure_diagnosis": list(FAILURE_MODES)` with no
definitions (`scripts/run_retrace_bench_baseline.py:492`). The model must guess
label semantics. Predictable confusions (seen in §2.4 matrices):
`under_update` vs `stale_memory_reuse`; `over_update` vs `scope_leakage`;
`failure_to_forget` vs `stale_memory_reuse`; `failure_to_release_or_restore` vs
`conflict_collapse`; `memory_hallucination` vs `scope_leakage`/`wrong_source`;
`wrong_source_attribution` vs `scope_leakage`.

### 3.6 Metric naming ambiguity (predicted-frequency vs accuracy)
In `benchmark/retrace_bench/scorers_general.py`, the per-mode loop (`:264-265`)
sets `<mode>_rate = (gold == mode AND pred == mode)` (a true-positive indicator),
but `:266-269` then **overwrite** four of them to pure predicted frequency:
`under_update_rate`, `over_update_rate`, `scope_leakage_rate`,
`policy_violation_rate` = `float(pred == mode)`. So `scope_leakage_rate = 0.45`
means "45% of predictions were `scope_leakage`", which is **not** comparable to
e.g. `stale_memory_reuse_rate = 0.055` (a TP indicator). The names invite
misreading these as per-mode accuracy.

**Recommendation (non-destructive):** keep current keys for backward
compatibility but add clearly named aliases — `predicted_under_update_rate`,
`predicted_over_update_rate`, `predicted_scope_leakage_rate`,
`predicted_policy_violation_rate` — and document the gold∧pred TP-indicator
semantics of the remaining `<mode>_rate` keys. Do **not** silently rewrite the
historical `*.metrics.json` outputs.

### 3.7 Strict paraphrase scoring for `answer_key_fact_accuracy`
`rubric["must_include"] = [texts["current"]]` (`generator:549`) is a whole
sentence; `key_fact_matches()` requires ≥0.80 token overlap of each entry. On
800: **every** `must_include` entry is 9–20 words (mean 12.8). A correct
paraphrase that drops/reorders tokens fails. This is why
`answer_key_fact_accuracy` is only 0.425/0.545 even when decisions are correct
(`dec_ok_keyfact_zero` appears on many suspicious scenarios). Auxiliary-metric
risk; v2 should use atomic key facts.

---

## 4. Two-model evidence tables

### 4.1 Global metrics (first 200)

| metric | deepseek_v3 | glm_51 | max gap |
| --- | --- | --- | --- |
| black_box_decision_accuracy | 0.845 | 0.920 | 0.075 |
| decision_macro_f1 | 0.760 | 0.897 | 0.136 |
| decision_balanced_accuracy | 0.762 | 0.880 | 0.118 |
| non_answer_decision_accuracy | 0.777 | 0.892 | 0.115 |
| use_current_memory_accuracy | 1.000 | 0.984 | 0.016 |
| memory_state_accuracy | 0.746 | 0.807 | 0.061 |
| evidence_f1 | 0.714 | 0.639 | 0.075 |
| failure_diagnosis_accuracy | 0.310 | 0.395 | 0.085 |
| stale_reuse_rate | 0.060 | 0.095 | 0.035 |
| answer_key_fact_accuracy | 0.425 | 0.545 | 0.120 |
| answer_exact_match | 0.055 | 0.025 | 0.030 |
| format_failure_rate | 0.000 | 0.000 | 0.000 |
| forbidden_fact_hits | 0.115 | 0.145 | 0.030 |

### 4.2 Per-expected-decision accuracy

| expected_decision | count | deepseek_v3 | glm_51 | gap | flag |
| --- | --- | --- | --- | --- | --- |
| use_current_memory | 61 | 1.000 | 0.984 | 0.016 | TOO_EASY (≥0.95 both) |
| escalate | 34 | 1.000 | 1.000 | 0.000 | TOO_EASY (≥0.95 both) |
| ask_clarification | 45 | 0.711 | 0.867 | 0.156 | DISCRIMINATIVE |
| refuse_due_to_policy | 20 | 0.100 | 0.550 | 0.450 | DISCRIMINATIVE |
| mark_unresolved | 40 | 1.000 | 1.000 | 0.000 | TOO_EASY (≥0.95 both) |

### 4.3 Per-primary-failure-mode highlights (diag_acc = failure_diagnosis_accuracy)

| failure_mode | model | decision_acc | memory_state_acc | evidence_f1 | diag_acc | flags |
| --- | --- | --- | --- | --- | --- | --- |
| stale_memory_reuse | ds_v3 / glm | 1.000 / 1.000 | 0.829 / 0.908 | 0.758 / 0.684 | 0.579 / 0.579 | DECISION_SATURATED |
| under_update | ds_v3 / glm | 0.895 / 1.000 | 0.779 / 0.868 | 0.719 / 0.696 | **0.000** / 0.105 | DIAG_LOW_BOTH |
| over_update | ds_v3 / glm | 1.000 / 1.000 | 0.892 / 0.819 | 0.815 / 0.657 | **0.000** / 0.167 | DECISION_SATURATED |
| conflict_collapse | ds_v3 / glm | 1.000 / 0.889 | 0.725 / 0.900 | 0.828 / 0.648 | 0.833 / 0.722 | |
| scope_leakage | ds_v3 / glm | 1.000 / 1.000 | 0.725 / 0.836 | 0.741 / 0.713 | 0.778 / 0.778 | DECISION_SATURATED |
| policy_violation | ds_v3 / glm | 0.722 / 0.833 | 0.575 / 0.689 | 0.717 / 0.606 | 0.333 / 0.500 | |
| wrong_source_attribution | ds_v3 / glm | 0.722 / 1.000 | 0.714 / 0.881 | 0.680 / 0.643 | 0.611 / 0.889 | |
| memory_hallucination | ds_v3 / glm | 0.833 / 0.889 | 0.742 / 0.811 | 0.548 / 0.541 | **0.000** / 0.056 | DIAG_LOW_BOTH |
| unnecessary_memory_write | ds_v3 / glm | 0.389 / 0.778 | 0.686 / 0.817 | 0.433 / 0.498 | 0.111 / 0.444 | |
| failure_to_forget | ds_v3 / glm | 0.722 / 0.722 | 0.906 / 0.711 | 0.720 / 0.713 | 0.167 / 0.111 | |
| failure_to_release_or_restore | ds_v3 / glm | 1.000 / 1.000 | 0.625 / 0.628 | 0.889 / 0.620 | **0.000** / **0.000** | DIAG_ZERO_BOTH;DECISION_SATURATED |

### 4.4 Predicted-diagnosis distribution (over-prediction of `scope_leakage`)

| predicted label | deepseek_v3 | glm_51 |
| --- | --- | --- |
| scope_leakage | **90** | **64** |
| stale_memory_reuse | 39 | 53 |
| conflict_collapse | 36 | 32 |
| wrong_source_attribution | 15 | 24 |
| policy_violation | 14 | 9 |
| unnecessary_memory_write | 3 | 8 |
| over_update | 0 | 5 |
| failure_to_forget | 3 | 2 |
| under_update | 0 | 2 |
| memory_hallucination | 0 | 1 |
| failure_to_release_or_restore | 0 | 0 |

`under_update`, `over_update`, `memory_hallucination`, and
`failure_to_release_or_restore` are **almost never predicted** by either model.

### 4.5 Pairwise (200 common scenarios)

- both decision correct: **166**
- both diagnosis wrong: **109**
- both predicted the **same** wrong diagnosis: **77**
- both `evidence_f1 < 0.5`: 9
- both `memory_state_accuracy < 0.75`: 26
- both over-predicted `scope_leakage` when gold ≠ `scope_leakage`: **38**

---

## 5. Top suspicious scenarios (excerpt)

Full list (top 30) is regenerated by `scripts/audit_retrace_bench_outputs.py`.
Representative rows (`dec` and `diag` shown as DeepSeek-V3 / GLM):

| scenario_id | domain | gold_mode | gold_dec | dec (A/B) | diag (A/B) | reasons |
| --- | --- | --- | --- | --- | --- | --- |
| rt-…-000028 | calendar_task_workflow | policy_violation | use_current_memory | uc / uc | scope_leakage / scope_leakage | dec_ok_diag_wrong, overpred_scope, ev_low, keyfact_zero |
| rt-…-000006 | personal_assistant_preference | policy_violation | refuse_due_to_policy | uc / uc | scope_leakage / scope_leakage | both_fail_refuse, overpred_scope |
| rt-…-000010 | enterprise_multi_tool_workflow | failure_to_forget | use_current_memory | uc / uc | scope_leakage / scope_leakage | dec_ok_diag_wrong, overpred_scope, keyfact_zero |
| rt-…-000022 | personal_assistant_preference | failure_to_release_or_restore | mark_unresolved | mu / mu | conflict_collapse / conflict_collapse | dec_ok_diag_wrong, keyfact_zero |
| rt-…-000146 | enterprise_multi_tool_workflow | over_update | mark_unresolved | mu / mu | scope_leakage / scope_leakage | dec_ok_diag_wrong, overpred_scope, keyfact_zero |

Pattern: **decision correct (often because the action word leaks) but diagnosis
collapses to `scope_leakage` or `conflict_collapse`, and the whole-sentence
key-fact rubric scores 0** even though the answer is right.

---

## 6. Recommendation

1. **Do not present v1 `failure_diagnosis_accuracy` or saturated decision
   classes as headline claims.** Move `failure_diagnosis_accuracy` from
   `HEADLINE_METRICS` to a diagnostic/analysis metric; keep
   `decision_macro_f1` / `non_answer_decision_accuracy` but footnote that
   `escalate` / `mark_unresolved` are currently leakage-inflated in v1.
2. **Add clearer predicted-frequency aliases** in the scorer
   (`predicted_scope_leakage_rate`, etc.) without rewriting historical outputs.
3. **Build `templateheldout_v2`** before a final paper-facing release that:
   de-actionalizes authoritative records (state, not verbs); localizes the
   diagnostic query to a concrete `event_id`/`memory_id`; makes cross-scope
   distractors conditional (universal only for `scope_leakage`); varies evidence
   source labels; uses atomic-fact rubrics; and ships failure-mode definitions to
   the baseline prompt.
4. **Preserve v1** (`test_800_templateheldout_en`, `test_800_en`) as
   prototype/diagnostic. Do not mutate or overwrite it.

This audit does **not** conclude the benchmark is useless or perfect: the
state/evidence/policy axes are working and model-separating; the decision and
diagnosis axes need the v2 fixes above before they carry headline weight.

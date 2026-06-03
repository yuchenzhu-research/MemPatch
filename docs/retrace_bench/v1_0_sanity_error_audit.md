# ReTrace-Bench v1.0 — sanity pilot error audit

> **Pre-full-run audit.** Diagnostic analysis of the v1.0 sanity pilot
> (`docs/retrace_bench/v1_0_sanity_model_pilot.md`) on small subsets, run before
> launching expensive full jobs. No dataset, generator, or scorer changes were
> made. Subsets: `hard_300_en` first 50, `main_3000_en` first 100; models
> `Pro/moonshotai/Kimi-K2.6`, `Pro/zai-org/GLM-5.1`, `deepseek-ai/DeepSeek-V4-Pro`,
> baseline `llm_json_answerer`. `realistic_100_en` excluded (annotation pending).

## 1. hard_300 first50 — failure-diagnosis collapse

### 1.1 Predicted-diagnosis distribution (50 cases / model)

The models do **not** use the 11-way taxonomy. They collapse onto 2–4 labels,
overwhelmingly `wrong_source_attribution`:

| model | predicted label distribution (count / 50) |
| --- | --- |
| Kimi-K2.6 | wrong_source_attribution 37, scope_leakage 13 |
| GLM-5.1 | wrong_source_attribution 45, scope_leakage 5 |
| DeepSeek-V4-Pro | wrong_source_attribution 23, scope_leakage 15, policy_violation 6, conflict_collapse 6 |

Raw model output is well-formed (single valid enum strings, `format_failure_rate
= 0.000`), so this is a genuine labeling behavior, not a parse/normalization
artifact — the prompt already enumerates all 11 enums plus
`failure_mode_definitions`.

### 1.2 Per-mode diagnosis accuracy (gold = `primary_failure_mode` == `expected_failure_diagnosis`)

| gold failure mode | Kimi | GLM | DeepSeek | mean | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| stale_memory_reuse | 0.00 | 0.00 | 0.00 | 0.00 | 5 |
| under_update | 0.00 | 0.00 | 0.00 | 0.00 | 5 |
| over_update | 0.00 | 0.00 | 0.00 | 0.00 | 5 |
| conflict_collapse | 0.00 | 0.00 | 0.40 | 0.13 | 5 |
| scope_leakage | 0.00 | 0.20 | 0.40 | 0.20 | 5 |
| policy_violation | 0.00 | 0.00 | 0.60 | 0.20 | 5 |
| wrong_source_attribution | 0.75 | 1.00 | 0.25 | 0.67 | 4 |
| memory_hallucination | 0.00 | 0.00 | 0.00 | 0.00 | 4 |
| unnecessary_memory_write | 0.00 | 0.00 | 0.00 | 0.00 | 4 |
| failure_to_forget | 0.00 | 0.00 | 0.00 | 0.00 | 4 |
| failure_to_release_or_restore | 0.00 | 0.00 | 0.00 | 0.00 | 4 |

**Near-zero (≤0.10 mean across all three models):** `stale_memory_reuse`,
`under_update`, `over_update`, `memory_hallucination`,
`unnecessary_memory_write`, `failure_to_forget`, `failure_to_release_or_restore`
(7 of 11 modes).

The only "good" cell, `wrong_source_attribution` (mean 0.67), is an artifact of
the collapse: models emit that label for almost everything, so it scores high
exactly where gold happens to be `wrong_source_attribution`. It reflects a
default, not diagnostic skill.

### 1.3 Confusion pattern

Across every gold mode, the top predicted label is `wrong_source_attribution`
(secondarily `scope_leakage`). Example rows (Kimi-K2.6):

```
stale_memory_reuse  -> wrong_source_attribution:3, scope_leakage:2
under_update        -> wrong_source_attribution:4, scope_leakage:1
memory_hallucination-> wrong_source_attribution:4
failure_to_forget   -> wrong_source_attribution:3, scope_leakage:1
```

GLM-5.1 is the most collapsed (45/50 → wrong_source_attribution); DeepSeek-V4-Pro
spreads slightly more (4 labels) and is the only model with any signal on
`conflict_collapse`/`policy_violation`.

### 1.4 High-evidence / wrong-diagnosis cases (10 representative)

70 of 150 (model × case) rows have `evidence_f1 ≥ 0.9` **and** a wrong diagnosis
— i.e. the model retrieved the right evidence events and (usually) chose the
right decision, but still mislabeled the failure category. Representative cases
(all `evidence_f1 = 1.00`):

| # | scenario | model | gold diag | pred diag | gold decision | in secondary? |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | rb-hard-300-en-00001 | Kimi | stale_memory_reuse | scope_leakage | use_current_memory | no |
| 2 | rb-hard-300-en-00024 | Kimi | under_update | wrong_source_attribution | ask_clarification | **yes** |
| 3 | rb-hard-300-en-00003 | Kimi | over_update | wrong_source_attribution | use_current_memory | no |
| 4 | rb-hard-300-en-00008 | Kimi | memory_hallucination | wrong_source_attribution | ask_clarification | no |
| 5 | rb-hard-300-en-00009 | Kimi | unnecessary_memory_write | wrong_source_attribution | use_current_memory | no |
| 6 | rb-hard-300-en-00010 | Kimi | failure_to_forget | wrong_source_attribution | use_current_memory | no |
| 7 | rb-hard-300-en-00011 | Kimi | failure_to_release_or_restore | scope_leakage | use_current_memory | no |
| 8 | rb-hard-300-en-00026 | GLM | conflict_collapse | wrong_source_attribution | ask_clarification | **yes** |
| 9 | rb-hard-300-en-00038 | Kimi | scope_leakage | wrong_source_attribution | mark_unresolved | no |
| 10 | rb-hard-300-en-00006 | Kimi | policy_violation | scope_leakage | refuse_due_to_policy | no |

### 1.5 Verdict: genuine difficulty vs taxonomy/generator ambiguity — **mostly genuine model difficulty, with a real but secondary ambiguity component**

Evidence for **genuine fine-grained difficulty / model behavior**:
- The collapse holds even when the model gets the decision and evidence right
  (cases above), so it is not a retrieval failure — it is the fine-grained
  11-way labeling step that fails.
- Models default to one broad label (`wrong_source_attribution`) rather than
  distributing across the taxonomy, despite being given all enums + definitions.
- Diagnosis accuracy is low both on no-failure cases (`use_current_memory`,
  0.08) and on non-answer cases (0.12) — the weakness is not confined to one
  regime.

Evidence for **taxonomy / scoring ambiguity (real but minority)**:
- Scoring is single-label exact match against `primary_failure_mode`, yet
  scenarios carry `secondary_failure_modes`. **18% (24/134)** of wrong
  predictions are actually one of the scenario's *labeled secondary modes*
  (e.g. case 2, case 8) — defensible, but scored 0.
- `wrong_source_attribution`'s definition ("trusts a forwarded/non-authoritative
  record … or attributes a fact to the wrong source") semantically overlaps with
  `stale_memory_reuse`, `under_update`, `scope_leakage`, and `conflict_collapse`
  (all involve relying on a wrong/old/out-of-scope record), which makes it an
  attractive catch-all.
- Diagnosis is required even on `use_current_memory` scenarios where no failure
  is actually executed (17/50 here), making the label a latent-trap category
  rather than an observed error — inherently more ambiguous.

**Net:** the near-zero per-mode numbers are dominated by genuine model
difficulty with fine-grained diagnosis, amplified by a measurable (~18%)
single-label-vs-co-present-mode scoring ambiguity. This does **not** indicate a
broken dataset, but it does mean `failure_diagnosis_accuracy` (single-label exact
match) is **not yet a trustworthy headline metric** and should be reported with
care.

## 2. main_3000 first100 — decision vs structured-revision gap

### 2.1 Headline gap

Gold decision distribution is reasonably balanced (use_current_memory 31,
ask_clarification 23, mark_unresolved 20, escalate 16, refuse_due_to_policy 10),
so decision scores are meaningful — but structured channels lag badly:

| model | decision acc | memory_state | evidence_f1 | diagnosis |
| --- | ---: | ---: | ---: | ---: |
| Kimi-K2.6 | 0.840 | 0.754 | 0.509 | 0.550 |
| GLM-5.1 | 0.960 | 0.755 | 0.677 | 0.540 |
| DeepSeek-V4-Pro | 0.780 | 0.721 | 0.711 | 0.630 |

### 2.2 Correct decision but wrong structure

Restricting to cases where the **decision is correct**, almost all still have at
least one structured channel wrong:

| model | correct-decision cases | any structure wrong | mem wrong | evidence wrong | diag wrong | all-4 correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kimi-K2.6 | 84 | 84 (100%) | 60 | 83 | 37 | 0 |
| GLM-5.1 | 96 | 93 (97%) | 66 | 83 | 43 | 3 |
| DeepSeek-V4-Pro | 78 | 75 (96%) | 55 | 56 | 31 | 3 |

Joint reliability (decision correct **and** memory_state = 1.0 **and**
evidence_f1 = 1.0 **and** diagnosis correct) is essentially zero: **0/100,
3/100, 3/100**.

### 2.3 Verdict

Coarse decision accuracy **substantially overestimates** structured revision
reliability. A model can pick the right action (e.g. `ask_clarification`) ~84–96%
of the time while almost never producing a fully correct memory-state +
evidence + diagnosis bundle. Evidence F1 is the single largest contributor to
the gap, followed by memory_state; diagnosis compounds it (see §1). Headline
claims must lead with the structured metrics (or a joint all-correct rate), not
decision accuracy alone.

## 3. Recommendations (no dataset changes made)

- **Do not** treat `failure_diagnosis_accuracy` (single-label exact match) as a
  headline metric yet. Before the full suite, consider — as analysis-only, not a
  dataset edit — reporting it as (a) credit-if-in-{primary ∪ secondary}, and/or
  (b) a coarser grouping, alongside the strict number, so the ~18% co-present-mode
  ambiguity is visible. Keep the strict metric too.
- **Lead reporting with structured metrics** (memory_state, evidence_f1,
  diagnosis) and a joint "all-correct" rate; present decision accuracy as a
  coarse upper bound only.

### Go / no-go on scaling up

| run | recommendation | rationale |
| --- | --- | --- |
| **hard_300 full (300)** | **Proceed** | Harness is clean (0% format failure, 0 errors); the full split is needed to get stable per-mode (≈27/mode) diagnosis estimates and confirm the collapse generalizes. Cheap relative to main. |
| **main_3000 first500** | **Proceed** | Natural next step to confirm the decision-vs-structured gap and metric stability at larger n before committing to the full run. Recommended as the gating step. |
| **main_3000 full (3000)** | **Hold until after first500** | No blocking dataset issue found, but launch the full 3000 only after first500 confirms metrics are stable and the diagnosis-reporting decision (above) is settled. DeepSeek-V4-Pro is the throughput bottleneck (~13 s/case), so full 3000 × 3 models is the expensive job — gate it on the cheaper checks first. |

**Bottom line:** nothing here blocks scaling, and no dataset fix is warranted yet.
The diagnosis collapse is mostly real model difficulty (with a measurable
single-label scoring ambiguity), and decision accuracy overstates structured
reliability — both are reporting/metric-framing issues to settle before the full
main_3000 run, not generation bugs.

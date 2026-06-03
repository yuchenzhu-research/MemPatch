# ReTrace-Bench v1.0 — Gated Model Pilot

Gated scaling step **before** any full `main_3000` run. It scales the sanity
pilot to (a) `hard_300_en` **full 300** and (b) `main_3000_en` **first 500**, for
three SiliconFlow models, to check metric stability and re-test the audit thesis
at larger `n`:

> **Coarse decision accuracy substantially overestimates structured memory
> revision reliability.**

This is still a pilot; it does not replace the full `main_3000` table and does
not over-claim. No dataset/generator/scorer/schema/HF changes were made.

---

## 1. Run configuration

| Field | Value |
| --- | --- |
| Branch | `benchmark` |
| Commit (HEAD at run time) | `54dc55d` |
| Provider | `siliconflow` (OpenAI-compatible) |
| Baseline | `llm_json_answerer` |
| Generation | `--max-tokens 1024 --disable-thinking` |
| Models | `Pro/moonshotai/Kimi-K2.6`, `Pro/zai-org/GLM-5.1`, `deepseek-ai/DeepSeek-V4-Pro` |
| Split A | `hard_300_en` — **full 300 cases** |
| Split B | `main_3000_en` — **first 500 cases** |
| Resume | `--resume` (idempotent; per-case append + incremental `.metrics.json`) |

Notes:
- `realistic_100_en` is **excluded** — annotation is still pending; it is not
  treated as official.
- `main_3000` **full (3000)** is **not** run here. This pilot gates that decision.
- All six runs completed `rc=0` with **`format_failure_rate = 0.000`** and no
  dropped/errored/partial cases (counts exact: 300/300/300, 500/500/500).

Outputs: `outputs/retrace_bench/v1_0/hard_300/`,
`outputs/retrace_bench/v1_0/main_3000_first500/`. Machine-readable aggregates:
`outputs/retrace_bench/v1_0/gated_summary.{json,jsonl}`. Reproduce with the
read-only `scripts/analyze_retrace_v1_gated_results.py`.

---

## 2. Six primary metrics

`fmt` = format_failure_rate, `decF1` = decision_macro_f1,
`nonAns` = non_answer_decision_accuracy, `mem` = memory_state_accuracy,
`evF1` = evidence_f1, `diag` = failure_diagnosis_accuracy (strict, official),
`stale` = stale_reuse_rate.

### hard_300_en — full 300

| model | n | fmt | decF1 | nonAns | mem | evF1 | diag | stale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kimi-K2.6 | 300 | 0.000 | 0.783 | 0.686 | 0.571 | 0.897 | 0.097 | 0.047 |
| GLM-5.1 | 300 | 0.000 | 0.933 | 0.971 | 0.713 | 0.875 | 0.097 | 0.007 |
| DeepSeek-V4-Pro | 300 | 0.000 | 0.768 | 0.786 | 0.670 | 0.889 | 0.150 | 0.033 |

### main_3000_en — first 500

| model | n | fmt | decF1 | nonAns | mem | evF1 | diag | stale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kimi-K2.6 | 500 | 0.000 | 0.767 | 0.716 | 0.754 | 0.515 | 0.534 | 0.086 |
| GLM-5.1 | 500 | 0.000 | 0.934 | 0.955 | 0.755 | 0.672 | 0.516 | 0.052 |
| DeepSeek-V4-Pro | 500 | 0.000 | 0.780 | 0.778 | 0.719 | 0.711 | 0.588 | 0.106 |

---

## 3. Decision-vs-structured gap (the key benchmark story)

`dec_acc` = black-box decision accuracy (share of cases with correct decision,
alias-aware). Gaps are `decF1 − structured_metric` (positive ⇒ decision score
is more optimistic than the structured channel).

### hard_300_en — full 300

| model | dec_acc | decF1 | mem | evF1 | diag | gap(mem) | gap(evF1) | gap(diag) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kimi-K2.6 | 0.780 | 0.783 | 0.571 | 0.897 | 0.097 | +0.212 | −0.114 | +0.686 |
| GLM-5.1 | 0.943 | 0.933 | 0.713 | 0.875 | 0.097 | +0.220 | +0.058 | +0.836 |
| DeepSeek-V4-Pro | 0.793 | 0.768 | 0.670 | 0.889 | 0.150 | +0.098 | −0.121 | +0.618 |

### main_3000_en — first 500

| model | dec_acc | decF1 | mem | evF1 | diag | gap(mem) | gap(evF1) | gap(diag) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kimi-K2.6 | 0.782 | 0.767 | 0.754 | 0.515 | 0.534 | +0.013 | +0.252 | +0.233 |
| GLM-5.1 | 0.944 | 0.934 | 0.755 | 0.672 | 0.516 | +0.179 | +0.262 | +0.418 |
| DeepSeek-V4-Pro | 0.808 | 0.780 | 0.719 | 0.711 | 0.588 | +0.061 | +0.069 | +0.192 |

### joint_all_correct_rate

A case is **joint-all-correct** only if **all four** hold: decision correct (`black_box_decision_accuracy == 1.0`), `memory_state_accuracy == 1.0`, `evidence_f1 == 1.0`, and strict `failure_diagnosis_accuracy == 1.0`.

| model / split | joint count | rate (all cases) | correct-decision cases | rate (among correct-decision) |
| --- | ---: | ---: | ---: | ---: |
| Kimi-K2.6 — hard_300 | 0 / 300 | **0.000** | 234 | 0.000 |
| GLM-5.1 — hard_300 | 0 / 300 | **0.000** | 283 | 0.000 |
| DeepSeek-V4-Pro — hard_300 | 1 / 300 | **0.003** | 238 | 0.004 |
| Kimi-K2.6 — main500 | 0 / 500 | **0.000** | 391 | 0.000 |
| GLM-5.1 — main500 | 17 / 500 | **0.034** | 472 | 0.036 |
| DeepSeek-V4-Pro — main500 | 27 / 500 | **0.054** | 404 | 0.067 |

**Confirmed at larger n.** Decision accuracy of 0.78–0.94 coexists with
joint-all-correct rates of **0.0–5.4%**. Even restricting to the 234–472 cases
per run where the decision is already correct, the fully-correct structured
revision rate is **≤ 6.7%**. Coarse decision accuracy is not a proxy for
structured memory-revision reliability.

---

## 4. Correct decision but structured failure

Among cases with a **correct decision**, how many still fail ≥1 structured
channel (memory / evidence / diagnosis), and how many get *all* structured
channels right.

### hard_300_en — full 300

| model | correct-decision | any struct wrong | mem wrong | evidence wrong | diagnosis wrong | all struct correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kimi-K2.6 | 234 | 234 (100%) | 232 | 103 | 209 | 0 |
| GLM-5.1 | 283 | 283 (100%) | 257 | 154 | 254 | 0 |
| DeepSeek-V4-Pro | 238 | 237 (99.6%) | 222 | 111 | 206 | 1 |

### main_3000_en — first 500

| model | correct-decision | any struct wrong | mem wrong | evidence wrong | diagnosis wrong | all struct correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kimi-K2.6 | 391 | 391 (100%) | 276 | 390 | 184 | 0 |
| GLM-5.1 | 472 | 455 (96.4%) | 327 | 412 | 232 | 17 |
| DeepSeek-V4-Pro | 404 | 377 (93.3%) | 279 | 285 | 180 | 27 |

The sanity-audit pattern **persists at larger n**: 93–100% of correct-decision
cases still have at least one structured channel wrong. The dominant failing
channel differs by split — diagnosis on `hard_300`, evidence on `main_3000` —
but in both, "right decision" rarely means "right revision."

---

## 5. Diagnosis analysis

Strict single-label `failure_diagnosis_accuracy` is the **official** metric.
The *primary-or-secondary* column below is **analysis-only** (credits a
prediction that matches the gold primary **or** any labeled secondary failure
mode); it is **not** a scoring change and is shown only to size label ambiguity.

| split | model | strict (official) | primary-or-secondary (analysis-only) |
| --- | --- | ---: | ---: |
| hard_300 | Kimi-K2.6 | 0.097 | 0.223 |
| hard_300 | GLM-5.1 | 0.097 | 0.263 |
| hard_300 | DeepSeek-V4-Pro | 0.150 | 0.260 |
| main500 | Kimi-K2.6 | 0.534 | 0.582 |
| main500 | GLM-5.1 | 0.516 | 0.560 |
| main500 | DeepSeek-V4-Pro | 0.588 | 0.634 |

### Predicted diagnosis distribution

- **hard_300 — collapsed.** Models emit 2–7 of 11 labels, dominated by
  `wrong_source_attribution`: Kimi 229/300, GLM 266/300, DeepSeek 125/300 (with
  `scope_leakage` second). GLM uses only 2 labels.
- **main_3000 — broad.** All three models use 10–11 labels with a realistic
  spread (e.g. DeepSeek: stale_memory_reuse 111, conflict_collapse 97,
  wrong_source_attribution 82, policy_violation 63, …). No single-label collapse.

### Per-failure-mode strict diagnosis accuracy (per model)

hard_300 (n≈27–28 per mode):

| gold mode | Kimi | GLM | DeepSeek |
| --- | ---: | ---: | ---: |
| wrong_source_attribution | 0.70 | 0.89 | 0.37 |
| scope_leakage | 0.19 | 0.19 | 0.30 |
| stale_memory_reuse | 0.18 | 0.00 | 0.11 |
| policy_violation | 0.00 | 0.00 | 0.52 |
| conflict_collapse | 0.00 | 0.00 | 0.26 |
| failure_to_forget | 0.00 | 0.00 | 0.11 |
| failure_to_release_or_restore | 0.00 | 0.00 | 0.00 |
| memory_hallucination | 0.00 | 0.00 | 0.00 |
| over_update | 0.00 | 0.00 | 0.00 |
| under_update | 0.00 | 0.00 | 0.00 |
| unnecessary_memory_write | 0.00 | 0.00 | 0.00 |

main_3000 first500 (n≈45–46 per mode):

| gold mode | Kimi | GLM | DeepSeek |
| --- | ---: | ---: | ---: |
| wrong_source_attribution | 0.93 | 1.00 | 0.82 |
| scope_leakage | 0.85 | 0.98 | 0.93 |
| policy_violation | 0.91 | 0.82 | 1.00 |
| stale_memory_reuse | 0.67 | 0.65 | 0.65 |
| unnecessary_memory_write | 0.67 | 0.69 | 0.62 |
| conflict_collapse | 0.59 | 0.33 | 0.85 |
| memory_hallucination | 0.64 | 0.07 | 0.33 |
| failure_to_forget | 0.24 | 0.24 | 0.76 |
| under_update | 0.30 | 0.24 | 0.22 |
| over_update | 0.00 | 0.54 | 0.07 |
| failure_to_release_or_restore | 0.07 | 0.11 | 0.22 |

### Gold → predicted confusion (hard_300, DeepSeek — most diverse predictor)

Almost every gold mode routes its plurality mass into
`wrong_source_attribution` / `scope_leakage`, e.g.:
`under_update → wrong_source_attribution (16/27)`,
`unnecessary_memory_write → wrong_source_attribution (15/27)`,
`failure_to_forget → wrong_source_attribution (13), scope_leakage (11)`. Only
`policy_violation → policy_violation (14/27)` keeps a correct plurality.

### Findings

- **`wrong_source_attribution` collapse persists — but only on `hard_300`.** On
  `main_3000` first500 the same taxonomy + scorer yield 0.82–1.00 strict
  accuracy for that mode and a broad label distribution. The collapse is a
  property of `hard_300`'s difficulty (adversarial L3/L4 cross-scope audits),
  **not** a global scorer/label defect.
- **Modes with near-zero strict accuracy across all three models on `hard_300`:**
  `failure_to_release_or_restore`, `memory_hallucination`, `over_update`,
  `under_update`, `unnecessary_memory_write` (all ≤ 0.10). On `main_3000`
  first500 **there are none** — every mode is diagnosed correctly some of the
  time by at least one model.
- **`scope_leakage` collapse does *not* reappear.** It is low on `hard_300`
  (0.19–0.30) but non-zero, and recovers strongly on `main_3000` (0.85–0.98).
  So the sanity-audit worry about a `scope_leakage`-specific collapse is not
  supported at scale.

No evidence of broken labels or invalid scoring. Strict single-label diagnosis
remains the official metric; report joint-all-correct alongside it as headline.

---

## 6. Per-decision analysis

Per-gold-decision count and decision accuracy.

### hard_300_en — full 300

| gold decision | n | Kimi | GLM | DeepSeek |
| --- | ---: | ---: | ---: | ---: |
| use_current_memory | 90 | 1.000 | 0.878 | 0.811 |
| escalate | 51 | 0.725 | 0.882 | **0.490** |
| ask_clarification | 70 | 0.643 | 1.000 | 0.914 |
| refuse_due_to_policy | 30 | 0.667 | 1.000 | 0.867 |
| mark_unresolved | 59 | 0.712 | 1.000 | 0.847 |

### main_3000_en — first 500

| gold decision | n | Kimi | GLM | DeepSeek |
| --- | ---: | ---: | ---: | ---: |
| use_current_memory | 148 | 0.939 | 0.919 | 0.878 |
| escalate | 86 | 0.977 | 0.930 | **0.453** |
| ask_clarification | 116 | 0.759 | 0.931 | 0.802 |
| refuse_due_to_policy | 49 | 0.878 | 0.959 | 0.837 |
| mark_unresolved | 101 | **0.366** | 1.000 | 1.000 |

Weak boundaries:
- **`escalate` recall is DeepSeek's worst class** (0.49 hard / 0.45 main) — it
  systematically under-escalates. GLM is strong on `escalate` (0.88–0.93).
- **`mark_unresolved` vs `escalate`:** Kimi collapses `mark_unresolved` on
  `main_3000` (0.366) while GLM/DeepSeek are perfect — a model-specific
  confusion of "unresolved" with "escalate/clarify," not a dataset issue.
- **`ask_clarification` vs `mark_unresolved`:** Kimi is the weakest clarifier
  (0.643 hard / 0.759 main).
- **`refuse_due_to_policy` vs `use_current_memory`:** generally well separated
  (0.67–1.00); GLM near-perfect.

---

## 7. Per-failure-mode structured metrics (averaged across the 3 models)

`dec` = decision accuracy, `mem` = memory_state_accuracy, `evF1` = evidence_f1,
`diag` = strict diagnosis accuracy, `stale` = stale_reuse_rate.

### hard_300_en — full 300

| gold mode | n | dec | mem | evF1 | diag | stale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conflict_collapse | 27 | 0.94 | 0.70 | 0.90 | 0.09 | 0.00 |
| failure_to_forget | 27 | 0.89 | 0.64 | 0.94 | 0.04 | 0.00 |
| failure_to_release_or_restore | 27 | 0.75 | 0.51 | 0.92 | 0.00 | 0.01 |
| memory_hallucination | 27 | 0.95 | 0.67 | 0.89 | 0.00 | 0.02 |
| over_update | 28 | 0.75 | 0.68 | 0.92 | 0.00 | 0.11 |
| policy_violation | 27 | 0.64 | 0.60 | 0.77 | 0.17 | 0.12 |
| scope_leakage | 27 | 0.73 | 0.59 | 0.78 | 0.22 | 0.04 |
| stale_memory_reuse | 28 | 0.99 | 0.78 | 0.91 | 0.10 | 0.00 |
| under_update | 28 | 0.88 | 0.68 | 0.92 | 0.00 | 0.01 |
| unnecessary_memory_write | 27 | 0.79 | 0.63 | 0.93 | 0.00 | 0.00 |
| wrong_source_attribution | 27 | 0.91 | 0.69 | 0.87 | 0.65 | 0.00 |

### main_3000_en — first 500

| gold mode | n | dec | mem | evF1 | diag | stale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conflict_collapse | 46 | 0.88 | 0.84 | 0.63 | 0.59 | 0.00 |
| failure_to_forget | 45 | 0.88 | 0.68 | 0.63 | 0.41 | 0.00 |
| failure_to_release_or_restore | 45 | 0.84 | 0.59 | 0.65 | 0.13 | 0.05 |
| memory_hallucination | 45 | 0.92 | 0.75 | 0.62 | 0.35 | 0.11 |
| over_update | 46 | 0.86 | 0.78 | 0.65 | 0.20 | 0.15 |
| policy_violation | 45 | 0.58 | 0.60 | 0.60 | 0.91 | 0.09 |
| scope_leakage | 46 | 0.84 | 0.74 | 0.64 | 0.92 | 0.16 |
| stale_memory_reuse | 46 | 0.88 | 0.90 | 0.66 | 0.66 | 0.22 |
| under_update | 46 | 0.92 | 0.68 | 0.60 | 0.25 | 0.07 |
| unnecessary_memory_write | 45 | 0.81 | 0.76 | 0.63 | 0.66 | 0.00 |
| wrong_source_attribution | 45 | 0.88 | 0.85 | 0.66 | 0.92 | 0.03 |

Observations: decision accuracy is high and roughly flat across modes, while
diagnosis is highly mode-dependent. `hard_300` keeps evidence_f1 high (0.77–0.94)
but crushes diagnosis; `main_3000` is the inverse (diagnosis recovers, evidence_f1
drops to ~0.60–0.66). The structured-channel difficulty is real and split-specific,
not concentrated in one broken mode.

---

## 8. Recommendation

**A. Proceed to `main_3000` full for all three models.**

Rationale:
- **Format failure:** `0.000` across all six runs (and all nine sanity runs).
  Harness, all three SiliconFlow providers, and the scorer are stable end-to-end.
- **Metric stability first100 → first500:** decision, memory, and evidence
  metrics are essentially unchanged from the sanity first100 subset; diagnosis on
  `main_3000` stays in the same band (sanity ~0.54–0.63 → first500 0.52–0.59).
  No regime change with more data.
- **`hard_300` full shows useful difficulty:** joint-all-correct ≈ 0 and 93–100%
  of correct-decision cases fail a structured channel — exactly the discriminative
  signal the benchmark is meant to expose, now stable at full `n=300`.
- **Diagnosis collapse is a *reporting* issue, not a dataset blocker:** it is
  confined to `hard_300`; on `main_3000` the identical taxonomy + scorer produce
  broad, sensible per-mode accuracy with no near-zero modes. There is **no
  evidence of broken labels or invalid scoring**, so no dataset change is
  warranted.
- **API cost / runtime (the only real caveat):** DeepSeek-V4-Pro is the
  bottleneck (~10–27 s/case vs ~3–10 s for Kimi/GLM). A full `main_3000` (3000 ×
  3 models) is dominated by DeepSeek and is many hours of wall-clock even fully
  parallelized across the three models. This is an operational cost, not a
  correctness concern.

Operational guidance for the full run:
- Run the three models **in parallel**, each `--resume` to its own file (this
  pilot did so safely; restart-resume was verified mid-run).
- Treat **strict** `failure_diagnosis_accuracy` as the official diagnosis metric
  and **lead with `joint_all_correct_rate`** as the headline reliability number;
  keep primary-or-secondary strictly as supplementary analysis.
- Expect `hard_300` diagnosis to stay low — report it as a difficulty result,
  not a harness failure.

Do **not** change the dataset, generators, scorer, schema, or HF package on the
basis of this pilot.

# ReTrace-Bench — Statistical Reporting Note

This note clarifies what is and is not reproducible about ReTrace-Bench numbers,
so the paper reports statistics honestly.

## 1. Generation seed vs model randomness

- **Dataset generation is deterministic.** `scripts/generate_retrace_bench_final.py`
  is driven by a single seed (`2027`). Re-running with the same seed reproduces
  every split byte-for-byte (same scenario/event/memory IDs, same gold). This is
  what the "seed" controls.
- **The generation seed is unrelated to model/API decoding randomness.** It does
  not influence, fix, or reproduce any model's outputs. A fixed dataset seed says
  nothing about model-result variance.

## 2. Temperature 0 is not exact vendor reproducibility

Even at `temperature=0`, hosted model APIs are **not** guaranteed to be
bit-reproducible: backend model versions, batching/kernel nondeterminism,
tokenization changes, and silent provider-side updates can all change outputs
over time. Therefore:
- Treat any model number as tied to a specific provider + model version + date.
- Re-running later may yield different scores even at temperature 0.
- Report the provider, model id, and run date alongside model results.

## 3. Bootstrap confidence intervals (no new API calls)

Confidence intervals over already-evaluated predictions are a **statistical
operation on existing artifacts** — they require no new model calls:
- For a metric computed per scenario, resample scenarios with replacement
  (e.g. 1000 bootstrap resamples), recompute the metric per resample, and report
  the 2.5th/97.5th percentiles as a 95% CI.
- This can be applied to any committed `*.jsonl` prediction set under
  `outputs/retrace_bench_v1_1/` (e.g. the offline baselines) without re-running
  anything.
- Bootstrap CIs quantify sampling variability over the **evaluated cases**; they
  do **not** capture provider-side model variance (see §2).

## 4. Multi-seed dataset generation (optional, future)

Regenerating the benchmark under additional seeds (e.g. 2028, 2029) and checking
that headline conclusions are stable is an **optional future robustness check**.
It is not part of this cleanup pass, and the canonical release is the single
seed-`2027` build.

## 5. What this cleanup pass did and did not measure

- **Did:** deterministic regeneration (seed 2027), full schema/leakage
  validation, gold-oracle self-consistency (1.0 on all core metrics), and offline
  rule-based baselines.
- **Did not:** run any paid/API model evaluation. No new model numbers were
  produced. Headline model comparison tables remain pending valid, committed
  prediction artifacts.
- **Lost-run caveat:** the previously lost hard500 API numbers (Kimi ≈ 0.054,
  GLM ≈ 0.140, DeepSeek ≈ 0.232 joint) are **internal lost-run approximations
  only**. They are not reproducible (no committed artifacts) and must not appear
  as official results or in any table.

## 6. Reporting checklist for the paper

- [ ] State the dataset seed (2027) and that splits are deterministic.
- [ ] For every model number: provider + model id + run date.
- [ ] Note temperature-0 ≠ exact reproducibility.
- [ ] Report bootstrap 95% CIs for headline metrics over committed predictions.
- [ ] Mark `realistic` as `synthetic_gold_unreviewed` until human-validated.
- [ ] Mark `calibration` as smoke/quickstart only (not for model selection).
- [ ] Keep lost-run hard500 numbers out of all official tables.

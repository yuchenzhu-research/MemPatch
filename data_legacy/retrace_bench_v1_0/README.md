# ReTrace-Bench v1.0 (DEPRECATED — legacy pilot)

> **ReTrace-Bench v1.0 is a legacy pilot release and is deprecated for new
> experiments. It is preserved for reproducibility of early development notes
> only. New evaluations should use the canonical ReTrace-Bench release prepared
> under v1.1.**

## Status

- **Deprecated / legacy.** Do **not** use these splits for new evaluations,
  baselines, leaderboard numbers, or paper claims.
- **Preserved, not deleted.** Kept for reproducibility of early development notes
  and provenance. See `docs/retrace_bench/v1_0_deprecation_notice.md`.
- **Not broken.** Deprecated means superseded, not defective — v1.0 simply
  predates the hardened, deterministic v1.1 construction and validation pipeline.
- **Do not mix v1.0 into canonical current evaluation.** Canonical results come
  only from the v1.1 splits in `data/retrace_bench_v1_1/`.

## Contents (legacy pilot splits)

| Split | Rows |
|---|---|
| `main_3000_en` | 3000 |
| `hard_500_en` | 500 |
| `hard_300_en` | 300 (early hard pilot) |
| `realistic_200_en` | 200 |
| `realistic_100_en` | 100 (early realistic pilot) |
| `calibration_80_en` | 80 |
| `private_hidden_200_en` | 200 (held-out; never published) |

## Canonical release

The canonical ReTrace-Bench (internally "v1.1") lives at
`data/retrace_bench_v1_1/` and is distributed on Hugging Face. See:
- `docs/retrace_bench/data_construction_statement.md`
- `docs/retrace_bench/v1_1_validation_report.md`
- `docs/retrace_bench/v1_1_hf_release_plan.md`

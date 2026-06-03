# ReTrace-Bench v1.0 — sanity model pilot

> **Sanity pilot, not the final paper table.** Small-subset smoke run of the
> three target models on the v1.0 splits to confirm the harness, providers,
> and scorers behave end-to-end. Numbers are on small subsets and must **not**
> be read as headline benchmark results. `realistic_100_en` is intentionally
> excluded (human annotation still pending).

## Run configuration

- Baseline: `llm_json_answerer`; provider: `siliconflow`; `--max-tokens 1024 --disable-thinking --resume`.
- Models: `Pro/moonshotai/Kimi-K2.6`, `Pro/zai-org/GLM-5.1`, `deepseek-ai/DeepSeek-V4-Pro`.
- Subsets: `calibration_80_en` full 80; `hard_300_en` first 50; `main_3000_en` first 100.
- Outputs: `outputs/retrace_bench/v1_0/sanity/<split_subset>/<model>.jsonl` (+ `.metrics.json`).

## Results

| model | split subset | n | format_failure_rate | decision_macro_f1 | non_answer_decision_accuracy | memory_state_accuracy | evidence_f1 | failure_diagnosis_accuracy | stale_reuse_rate | error/resumed cases |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kimi-K2.6 | calibration_80 (full 80) | 80 | 0.000 | 0.735 | 0.717 | 0.764 | 0.512 | 0.537 | 0.138 | 0 |
| GLM-5.1 | calibration_80 (full 80) | 80 | 0.000 | 0.949 | 0.962 | 0.764 | 0.692 | 0.575 | 0.050 | 0 |
| DeepSeek-V4-Pro | calibration_80 (full 80) | 80 | 0.000 | 0.747 | 0.755 | 0.719 | 0.722 | 0.562 | 0.113 | 0 |
| Kimi-K2.6 | hard_300 (first 50) | 50 | 0.000 | 0.843 | 0.788 | 0.583 | 0.893 | 0.060 | 0.000 | 0 |
| GLM-5.1 | hard_300 (first 50) | 50 | 0.000 | 0.947 | 0.970 | 0.726 | 0.875 | 0.100 | 0.000 | 0 |
| DeepSeek-V4-Pro | hard_300 (first 50) | 50 | 0.000 | 0.857 | 0.848 | 0.697 | 0.908 | 0.160 | 0.020 | 0 |
| Kimi-K2.6 | main_3000 (first 100) | 100 | 0.000 | 0.832 | 0.768 | 0.754 | 0.509 | 0.550 | 0.100 | 0 |
| GLM-5.1 | main_3000 (first 100) | 100 | 0.000 | 0.949 | 0.971 | 0.755 | 0.677 | 0.540 | 0.040 | 0 |
| DeepSeek-V4-Pro | main_3000 (first 100) | 100 | 0.000 | 0.730 | 0.739 | 0.721 | 0.711 | 0.630 | 0.110 | 0 |

Metric keys map directly to `all_metrics` in each `*.metrics.json`.

## Errors / failed / resumed cases

- All 9 runs completed with exit code 0; per-run prediction counts match the requested subset sizes (80 / 50 / 100).
- `format_failure_rate = 0.000` for every run, and no provider error-JSON responses were emitted (0 error cases across all runs).
- `--resume` was enabled; these were fresh runs (no pre-existing partial files), so nothing was skipped.

## Notes (do not over-interpret)

- This is a smoke/plumbing check on small subsets; absolute scores are not comparable to a full-suite paper table and should not be used for model selection or headline claims.
- `calibration_80` is smoke/quickstart only by design.
- The harness, all three SiliconFlow providers, and the general scorer ran cleanly end-to-end with zero format failures, which is the primary sanity signal.


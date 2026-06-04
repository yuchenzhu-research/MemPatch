# ReTrace-Bench — Repository Simplification Report

Inventory + classification of the repository after the canonical v1.1 release prep
(merge commit `c8bc5d7`). Goal: keep GitHub focused on code, evaluator/validator,
scripts, docs, manifests/checksums, dataset card, and small demos — not a dump of
full data, test outputs, and raw API logs. The full canonical dataset is
distributed on Hugging Face.

Branch at time of writing: a cleanup branch off `benchmark` (content identical to
merged `main`).

## Classification

### A. Core code to keep
- `benchmark/retrace_bench/` — scorers, api, taxonomy, public_view, generation,
  baselines, evaluation, protocols, utils, llm_providers.
- Core scripts: `validate_retrace_bench_dataset.py`,
  `check_retrace_bench_gold_oracle.py`, `evaluate_retrace_bench_predictions.py`,
  `run_retrace_bench_baseline.py`, `generate_retrace_bench_final.py`,
  `generate_retrace_bench_blueprints.py`, `render_retrace_bench_dataset.py`,
  `export_human_annotation_packet.py`, `score_human_annotations.py`,
  `build_hf_release_v1_1.py`, `package_hf_retrace_bench.py`,
  `check_retrace_split_leakage.py`.
- `tests/retrace_bench/**` and supporting `tests/**`.

### B. Canonical v1.1 release docs/manifests/checksums to keep
- `data/retrace_bench_v1_1/{main_3000_en,hard_500_en,realistic_200_en,calibration_80_en}/`
  (`scenarios.jsonl` + `manifest.json` + `README.md`).
- `hf_release/retrace_bench_v1_1/` metadata: `README.md`, `DATASET_LICENSE.md`,
  `manifest.json`, `checksums.json`, `dataset_info.json`, `VERSION`, per-split
  `manifest.json`.
- `outputs/retrace_bench_v1_1/gold_oracle/*.metrics.json` and
  `outputs/retrace_bench_v1_1/baselines/*.metrics.json` (small, reproducibility).
- Reports under `docs/retrace_bench/`: `v1_1_validation_report.md`,
  `v1_1_offline_baseline_report.md`, `statistical_reporting_note.md`,
  `v1_1_hf_release_plan.md`, `v1_1_cleanup_plan.md`, human-validation docs,
  and this report.

### C. Full dataset files that should live on HF, not GitHub
- `hf_release/retrace_bench_v1_1/<split>/scenarios.jsonl` (rebuilt locally via
  `scripts/build_hf_release_v1_1.py`; git-ignored). GitHub keeps only the
  metadata in (B). The committed `data/retrace_bench_v1_1/` splits are the
  paper-facing copy; the authoritative full distribution is the HF bundle.

### D. Legacy v1.0 deprecated assets to keep temporarily
- `data_legacy/retrace_bench_v1_0/**` — v1.0 pilot splits (preserved, deprecated).
- `outputs/retrace_bench/v1_0/**` — official v1.0 model-suite outputs (tracked per
  AGENTS.md; kept for provenance).
- v1.0 docs: `docs/retrace_bench/v1_0_*.md`, `dataset_card_hf.md` (legacy card),
  and older design/audit notes.

### E. Old smoke/dev artifacts safe to remove later
- `outputs/retrace_bench_hard50/`, `outputs/retrace_bench_hard150/` (pre-balanced
  dev runs), `outputs/retrace_bench_gemini_hard150/` (Gemini first-20 partial run),
  `outputs/retrace_bench_siliconflow_hard150{,_balanced,_smoke}/`,
  `outputs/retrace_bench_siliconflow_hard500_candidate/`,
  `outputs/retrace_bench_hard500_candidate/`, `outputs/retrace_bench_smoke/`.
- Action taken this pass: **untracked** from git (`git rm --cached`) and added to
  `.gitignore`. Local copies remain on disk; full history is preserved in git, so
  nothing is destroyed. They are no longer part of the GitHub working tree.

### F. Raw API/log/prediction files that should be ignored
- 300 raw API response files under
  `outputs/retrace_bench_hard{50,150}/api_models/*.raw/*.json`.
- Large regenerable prediction dumps:
  `outputs/retrace_bench_v1_1/baselines/*.jsonl` (~43MB; already git-ignored).
- Action: `.gitignore` excludes `*.raw/`, raw API logs, and the bulky baseline
  prediction JSONL; raw API responses untracked this pass.

### G. ReTrace-Learn assets that must not be touched
- `src/retrace_learn/**`, `src/retracemem/**`, `data/retrace_learn/**`,
  `experiments/multiagent/**`, `prompts/**` (method track). Untouched.

## Summary of conservative actions in this cleanup pass
- No dataset **content** changes (no scenario edits); generation logic frozen.
- v1.0 marked deprecated (notice + legacy README); not deleted.
- Old dev/smoke/API outputs (E/F) untracked + git-ignored (recoverable from
  history); official v1.0 + v1.1 metrics retained.
- `.gitignore` hardened against tokens, `.env`, raw API responses, HF caches,
  huge prediction dumps, and temporary smoke outputs.
- HF bundle preflight (validate + gold oracle + checksums); no publish.

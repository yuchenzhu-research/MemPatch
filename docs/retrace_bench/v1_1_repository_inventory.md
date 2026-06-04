# ReTrace-Bench v1.1 — Repository Inventory (Phase 1 Audit)

> **Status:** Phase 1 audit only. **Nothing is deleted or moved by this report.**
> It classifies the current repository state so a later cleanup (after the
> hard500 go/no-go decision) can be executed safely.
>
> **Scope:** ReTrace-Bench track only (`benchmark/retrace_bench/`,
> `data/retrace_bench*`, `outputs/retrace_bench*`, `docs/retrace_bench/`,
> benchmark `scripts/`, `tests/retrace_bench/`). The ReTrace-Learn track
> (`src/retrace_learn/`, `src/retracemem/`, `data/retrace_learn/`) is **out of
> scope and must not be modified.**
>
> Branch: `benchmark` · Audited commit: `0128061` ("harden public view,
> balanced hard150, hard500 candidate").

## 0. Terminology and a critical packaging finding

"v1.1" is the internal name for the **hardened** benchmark pipeline that is
already the code at HEAD (`0128061`): sanitized public model-facing views and
balanced expected-decision scheduling for hard splits. "v1.0" refers to the
earlier/legacy state (gated model pilot, pre-balanced hard50/hard150,
template-heldout experiments). The paper-facing name should remain
**"ReTrace-Bench"**; this inventory is internal-only and does not narrate a
v1.0→v1.1 public history.

**Critical finding — most `scenarios.jsonl` datasets are NOT tracked in git.**
`.gitignore` ignores `*.jsonl` globally and then allowlists **only** specific
paths. The allowlist covers `data/retrace_bench/<canonical split>/**` and
`release/huggingface/**`, but **not** the auxiliary candidate directories
`data/retrace_bench_hard150_balanced/`, `data/retrace_bench_hard500_candidate/`,
`data/retrace_bench_hard150/`, `data/retrace_bench_hard50/`, or
`data/retrace_bench_smoke/`. For those directories only `manifest.json` and
`README.md` are committed; the actual `scenarios.jsonl` exists only on the
machine that generated it and is **absent from a fresh clone.**

Consequences:
- The path the task calls the "current best candidate hard split"
  (`data/retrace_bench_hard150_balanced/hard_150_en/scenarios.jsonl`) **does not
  exist in a fresh clone.** Its eval outputs under
  `outputs/retrace_bench_siliconflow_hard150_balanced/` *are* tracked (because
  `outputs/**` is allowlisted), so the results are reproducible-by-record but
  the dataset itself is not version-controlled.
- The hard500 candidate dataset must be **regenerated deterministically**
  (seed `2027`) for Phase 2; the committed
  `data/retrace_bench_hard500_candidate/hard_500_en/manifest.json` +
  `outputs/retrace_bench_hard500_candidate/gold_oracle.metrics.json` are the
  only surviving artifacts of the prior generation.

This is the single most important thing to resolve in the cleanup plan: decide
which datasets are git-tracked vs. Hugging-Face-hosted, and make the allowlist
match that decision (see `v1_1_cleanup_plan.md`).

Secondary finding — **pre-existing test failures.** `pytest tests/retrace_bench`
at `0128061` reports **7 failing tests** (schema drift between committed
`data/retrace_bench/*` splits and the validator/test expectations, e.g.
`secondary_failure_modes`/`tasks` field mismatches, shared event ids across
splits, HF package README template). These are **pre-existing on the branch and
not introduced by this audit.** They are relevant to the cleanup plan but are
not fixed here.

---

## A. Keep as core code (canonical, evaluation-neutral)

These are the benchmark engine: schema, scorer, validator, public-view
sanitizer, baselines, API runners, generation blueprints/pattern spec, and the
minimal test suite. Keep all of these.

### A.1 Benchmark package — `benchmark/retrace_bench/`
| Path | Role |
| --- | --- |
| `schemas.py`, `schemas_v2.py`, `general_schema.py` | Scenario / prediction dataclasses & schema |
| `scorers_general.py`, `evaluation/scorers.py`, `evaluation/run_evaluation.py`, `evaluation/aggregate_results.py` | **Scorer** + metric aggregation |
| `validation_v2.py` | Dataset **validator** core |
| `public_view.py` | **Public view sanitizer** (`INTERNAL_ONLY_FIELDS`, `public_scenario_view`, `sanitize_public_input`) |
| `general_taxonomy.py`, `taxonomy_v2.py` | Decisions / memory states / failure modes / patterns / `canonical_hidden_gold_fields` |
| `generation/pattern_spec.py` | **Pattern spec** + balanced hard decision schedule (`HARD_DECISION_TARGET_FRACTIONS`, `build_hard_pattern_decision_plan`) |
| `generation/hard_plus_blueprints.py`, `generation/github_realistic_blueprints.py`, `generation/evidence_dependency_graph.py`, `generation/adversarial_distractors.py`, `generation/seed_scenarios.py`, `generation/render_queries.py`, `generation/expand_scenarios.py`, `generation/deactionalized.py`, `generation/release_manifest.py`, `generation/validate_generated.py`, `generation/github_workflow_seeds.py` | **Generation blueprints** |
| `baselines/latest_only.py`, `baselines/retrieve_all.py`, `baselines/prompt_proposer.py`, `baselines/directjudge.py`, `baselines/crud_memory.py` | Baselines (sanity + method) |
| `protocols/*.py` | Structured / raw / oracle protocols |
| `api.py`, `utils/*.py` | Public API + id/hashing/jsonl/contamination/splits helpers |
| `evaluation/judge_prompts.py` | LLM-judge prompt scaffolding |

### A.2 Core scripts — `scripts/`
| Path | Role |
| --- | --- |
| `generate_retrace_bench_final.py` | **Canonical generator** for all final splits (incl. `--only hard --count hard=N`) |
| `validate_retrace_bench_dataset.py` | **Dataset validator CLI** |
| `check_retrace_bench_gold_oracle.py` | **Gold-oracle check** (replays `hidden_gold`, must score 1.0) |
| `run_retrace_bench_baseline.py` | Baseline runner (`latest_only`, `retrieve_all`, etc.) |
| `run_retrace_bench_siliconflow.py` | **SiliconFlow three-model runner** (sanitized prompts) |
| `run_retrace_bench_api_models.py` | Generic API model runner |
| `evaluate_retrace_bench_predictions.py`, `evaluate.py` | Prediction evaluation CLIs |
| `check_retrace_split_leakage.py` | Cross-split leakage check |
| `render_retrace_bench_dataset.py`, `render_retrace_bench_final.py` | Public-view rendering |
| `package_hf_retrace_bench.py`, `upload_to_hf.py` | HF packaging/upload |
| `generate_retrace_bench_blueprints.py` | Blueprint generation helper |

### A.3 Minimal tests — `tests/retrace_bench/`
Keep the suite: `test_public_view.py`, `test_pattern_spec.py`,
`test_hard_decision_schedule.py`, `test_scorers_fix.py`, `test_validation.py`,
`test_schema.py`, `test_public_api.py`, `test_baseline_no_gold_leak.py`,
`test_latest_only_baseline.py`, `test_balanced_metrics.py`,
`test_contamination_guard.py`, `test_siliconflow_retry.py`,
`test_generation_smoke.py`. (See §0 secondary finding: a subset is currently
failing due to committed-split schema drift — track for repair, do not delete.)

### A.4 Core docs (schema / metrics / protocol authority)
`docs/retrace_bench/README.md`, `schema.md`, `schema_v2_proposal.md`,
`metrics_v2.md`, `protocols_v2.md`, `quality_gates_v2.md`,
`generation_and_audit_protocol.md`, `dataset_design.md`,
`memory_reliability_taxonomy.md`, `failure_modes.md`, `contamination_policy.md`,
`manual_validation_protocol.md`, `industrial_domains.md`.

---

## B. Keep as v1.1 canonical candidate

The hardened, balanced artifacts that define the v1.1 release candidate.

| Path | Tracked? | Notes |
| --- | --- | --- |
| `data/retrace_bench_hard150_balanced/hard_150_en/manifest.json`, `README.md` | yes | Current best hard split metadata (L3=75/L4=75; decisions 75/30/18/15/12) |
| `data/retrace_bench_hard150_balanced/hard_150_en/scenarios.jsonl` | **NO (gitignored)** | The actual best-candidate dataset — **not version-controlled** (see §0) |
| `outputs/retrace_bench_siliconflow_hard150_balanced/` | yes (20 files) | `summary.metrics.json`, `summary.md`, `model_comparison.md`, per-model metrics, `gold_oracle.metrics.json`, `balanced_validation_report.md`, baseline metrics. DeepSeek joint≈0.227, GLM≈0.193, Kimi≈0.133; format_failure_rate=0 |
| `data/retrace_bench_hard500_candidate/hard_500_en/manifest.json`, `README.md` | yes | hard500 candidate metadata (L3=250/L4=250; decisions 250/100/60/50/40; avg_required_evidence=1.69) |
| `data/retrace_bench_hard500_candidate/hard_500_en/scenarios.jsonl` | **NO (gitignored)** | Must be **regenerated** in Phase 2 (seed 2027) |
| `outputs/retrace_bench_hard500_candidate/gold_oracle.metrics.json` | yes | Prior gold-oracle pass for hard500 (pass=true, joint=1.0) |
| `data/retrace_bench/hard_500_en/` | yes (scenarios tracked) | Canonical `final_aaai` hard-500 split (allowlisted) — overlaps conceptually with the candidate dir; reconcile in cleanup |
| `data/retrace_bench/main_3000_en/`, `realistic_200_en/`, `calibration_80_en/`, `private_hidden_200_en/`, `realistic_100_en/`, `hard_300_en/` | yes | Canonical final split family (validator schema drift noted in §0) |
| `release/huggingface/ReTrace-Bench/**` | yes | HF release snapshot (main/hard500/realistic200/calibration80 jsonl) |
| `docs/retrace_bench/dataset_card_hf.md`, `split_leakage_report.md`, `manual_validation_report.md`, `manual_validation_sample_88.md` | yes | Validation/leakage/dataset-card reports for the canonical family |
| `papers/retrace_bench/**`, `RETRACE_BENCH.md` | yes | Paper workspace + track entrypoint |

---

## C. Archive or move to legacy (v1.0 / pre-balanced / partial runs)

Do **not** delete — these document the v1.0 pilot and pre-hardening history and
should become a legacy appendix. Recommend moving under a `legacy/` or
`docs/retrace_bench/legacy/` namespace once hard500 passes.

| Path | Why legacy |
| --- | --- |
| `docs/retrace_bench/v1_0_gated_model_pilot.md`, `v1_0_sanity_model_pilot.md`, `v1_0_sanity_error_audit.md` | v1.0 gated pilot docs/results |
| `docs/retrace_bench/templateheldout_v1_model_audit.md`, `templateheldout_v2_*.md`, `template_lookup_test_800_*.md`, `template_signature_report.md` | Pre-v1.1 template-heldout experiments |
| `docs/retrace_bench/baseline_results_test_800_*.md`, `baseline_results_sample_80_hard_en.md`, `baseline_suite_v2.md` | Old baseline-suite results (pre-balanced) |
| `docs/retrace_bench/design.md`, `design_v2_industrial.md`, `final_hardening_plan.md`, `final_hardening_triage.md`, `benchmark_paper.md` | Development-history design/hardening docs (keep as appendix, not main narrative) |
| `data/retrace_bench_hard150/hard_150_en/` (manifest+README only) | Old **pre-balanced** hard150 (superseded by `_balanced`) |
| `data/retrace_bench_hard50/hard_50_en/` (manifest+README only) | hard50 mini split (superseded) |
| `outputs/retrace_bench_hard150/` (160 files), `outputs/retrace_bench_hard50/` (162 files) | Pre-balanced hard150/hard50 eval dumps |
| `outputs/retrace_bench_siliconflow_hard150/` (12 files) | Pre-balanced SiliconFlow hard150 run (legacy comparison baseline referenced by the runner) |
| `outputs/retrace_bench_gemini_hard150/` (7 files) | **Gemini partial first-20** run (`gemini_metrics_partial_n20.json`, `progress.json`) — incomplete |
| `scripts/run_retrace_bench_gemini_hard150.py`, `gemini_api.py`, `check_gemini_api.py` | Gemini partial-run tooling |
| `scripts/run_retrace_bench_hard_scale_test.py`, `build_retrace_bench_hard_scale_summary.py`, `build_retrace_bench_hard50_summary.py`, `build_retrace_bench_hard150_balanced_report.py` | Scale-test/report builders tied to legacy split dirs |
| `scripts/analyze_retrace_v1_gated_results.py`, `analyze_retrace_template_signatures.py`, `run_template_lookup_baseline.py` | v1.0 gated / template-heldout analysis |

---

## D. Delete later / local-only (raw responses, predictions, errors, smoke)

Heavy machine-local run artifacts. They are currently **tracked** (because
`outputs/**` is allowlisted), which bloats the repo. Recommend removing from git
after the v1.1 release is frozen and relying on HF + summary metrics instead.
**Do not delete in Phase 1.**

| Path | Class |
| --- | --- |
| `outputs/**/*.raw_responses.jsonl` (7 files incl. SiliconFlow + Gemini) | Raw model responses (contain answer text) |
| `outputs/**/*.predictions.jsonl` (20) | Per-scenario prediction dumps |
| `outputs/**/*.errors.jsonl`, `outputs/retrace_bench_gemini_hard150/gemini_errors.jsonl` | API error logs (incl. `403 Model is private` for `moonshotai/Kimi-K2.6`) |
| `outputs/retrace_bench_hard50/api_run.log` | Raw API run log |
| `outputs/retrace_bench_siliconflow_hard150_smoke/` (6 files) | Smoke run |
| `data/retrace_bench_smoke/` (5 split dirs, manifest+README) | Temporary smoke splits |
| `data/retrace_bench_hard{50,150}/` `scenarios.jsonl` (local-only, gitignored) | Pre-hardening generated data |

> Keep the **summary** artifacts (`summary.metrics.json`, `summary.md`,
> `model_comparison.md`, `*.metrics.json`, `gold_oracle.metrics.json`,
> validation reports) even when raw dumps are pruned — they are the compact,
> shareable record of results.

---

## E. Never commit (secrets / sensitive)

| Item | Status now |
| --- | --- |
| `.env` (SiliconFlow API key on first line) | **Not present / not tracked** (gitignored). `.env.example` is the only committed template. Good. |
| API keys / tokens / credentials in any form | **None found** in tracked files (scanned for `.env`, `secret`, `api_key`, `.pem`, `credential`). |
| Full raw API logs containing **prompts** or sensitive metadata | Current `*.raw_responses.jsonl` store model **output** + `scenario_id` (not the full prompt), and prompts are built from the **sanitized** public view, so leakage risk is low — but treat as local-only (category D) and never add prompt-bearing logs. |

**Rule going forward:** the SiliconFlow runner already reads the key from
`.env`/`--env-file`, redacts it from error logs (`redact_key`), and builds
prompts from `public_scenario_view` (no `hidden_gold`/`metadata`/`is_distractor`/
`source_pointers`). Keep it that way; never write the key to `outputs/` or logs.

---

## Summary of recommended actions (for the post-go/no-go cleanup plan)

1. Decide tracked-vs-HF dataset policy and **fix `.gitignore` allowlist** so the
   canonical v1.1 datasets are either committed or clearly HF-only (no silently
   ignored "candidate" dirs).
2. Move category **C** docs/artifacts under a `legacy/` namespace; keep as
   appendix, not main paper narrative.
3. After the v1.1 release is frozen, prune category **D** raw/prediction/error
   dumps from git (retain summary metrics).
4. Repair or quarantine the 7 pre-existing `tests/retrace_bench` failures
   (committed-split schema drift) — tracked separately from this benchmark task.
5. Never commit `.env`/keys (already clean).

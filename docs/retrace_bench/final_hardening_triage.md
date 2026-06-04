# ReTrace-Bench Final Hardening Triage

**Reference commit (WIP scaffold):** `68d853c99a22de49382784deab061e3e79177a45`  
**Patch status:** schema/scorer/validator/generator consistency fixes applied; smoke re-validated locally.

## Status summary

Commit `68d853c` introduced a **WIP AAAI hardening scaffold**, not a trustworthy final release. The pre-existing generated trees under `data/retrace_bench/{main_3000_en,hard_500_en,realistic_200_en,...}` from that scaffold **must not be used** — they were built with broken schema/manifest stats.

This patch makes the **pipeline internally consistent** on smoke splits. Controlled synthetic splits (`main`, `hard`, `calibration`, `private_hidden`) are smoke-testable tonight. **`realistic` remains unreviewed** and is not headline-ready.

**Paper fallback:** `data/retrace_bench/v1_0/` (main/hard/realistic/calibration) remains the authoritative fallback dataset for headline tables until final hardening completes manual realistic review and full-generation gates.

---

## Blockers fixed in this patch

| Blocker | Fix |
|--------|-----|
| `hidden_gold` schema mismatch (`memory_states`, `failure_diagnosis`, `minimal_evidence_event_ids`) | Generator now emits canonical fields; scorer/validator read via `canonical_hidden_gold_fields()` with legacy fallbacks |
| Memory status label mismatch (`authorized`, `superseded`, …) | `GENERATOR_MEMORY_STATUS_ALIASES` maps to `MEMORY_STATUSES`; validator rejects non-canonical statuses |
| Manifest `avg_required_evidence_count = 0.0` | `release_manifest._evidence_count()` counts `expected_evidence_event_ids` or `minimal_evidence_event_ids` |
| Realistic auto-marked `reviewed` | Default `annotation_status = synthetic_gold_unreviewed`; packaging validator blocks unreviewed realistic in `--packaging-final` |
| Missing gold-oracle gate | Added `scripts/check_retrace_bench_gold_oracle.py` |
| Distractor event ID collisions across scenarios | Distractor IDs now scoped per `scenario_id` |
| Gold-oracle `joint_revision_success` failure | `expected_answer` now satisfies `rubric.must_include` |

---

## Smoke validation status (seed 2027)

Generated with:

```bash
PYTHONPATH=. python scripts/generate_retrace_bench_final.py --smoke --seed 2027 --out data/retrace_bench_smoke
```

| Split | Validator | Gold oracle | Manifest `avg_required_evidence_count` |
|-------|-----------|-------------|----------------------------------------|
| `main_30_en` | PASS (0 errors) | PASS (all 1.0 / stale 0.0) | 1.733 |
| `hard_30_en` | PASS (0 errors) | PASS | 1.733 |
| `realistic_20_en` | PASS (0 errors; 20 warnings: unreviewed) | not headline split | 1.733 |
| `calibration_20_en` | PASS | PASS | 1.733 |

Realistic warnings are intentional: every scenario reports `annotation_status='synthetic_gold_unreviewed'`.

---

## Gold-oracle metrics (`hard_30_en`)

All required expectations met:

- `decision_macro_f1` = 1.0
- `black_box_decision_accuracy` = 1.0
- `memory_state_accuracy` = 1.0
- `evidence_f1` = 1.0
- `minimal_evidence_exact_match` = 1.0
- `failure_diagnosis_accuracy` = 1.0
- `stale_reuse_rate` = 0.0
- `joint_revision_success` = 1.0

---

## Baseline metrics on hard smoke (`hard_30_en`, n=30)

| Metric | `latest_only` | `retrieve_all` |
|--------|---------------|----------------|
| `decision_macro_f1` | 0.212 | 0.212 |
| `black_box_decision_accuracy` | 0.733 | 0.733 |
| `memory_state_accuracy` | 0.511 | 0.511 |
| `evidence_f1` | 0.0 | 0.437 |
| `failure_diagnosis_accuracy` | 0.1 | 0.1 |
| `latest_event_shortcut_failure_rate` | 0.267 | 0.267 |

Baselines behave as expected sanity checks (majority-class decision bias; weak evidence on `latest_only`).

---

## Is full generation (3,780 scenarios) safe tonight?

**No — not yet.**

Safe after this patch:

- Regenerate **controlled synthetic** splits only (`main`, `hard`, `calibration`, `private_hidden`) once a larger smoke gate (e.g. 100+ per split) passes the same validator + gold-oracle checks.

Still blocked:

1. **`realistic_200_en`** — synthetic GitHub-derived gold is **unreviewed**; do not use for headline tables or claim human validation.
2. **Legacy WIP trees** from `68d853c` in `data/retrace_bench/` — invalid schema/stats; ignore or delete locally before any publish step (do **not** remove `v1_0/`).
3. **Schema v2 task layout** — final-hardening scenarios use top-level `black_box_task` fields instead of v1.0 `tasks` list; downstream packaging/tests expecting v1.0 row shape still fail until aligned.

---

## Recommendation: fixed hardening vs v1.0 fallback

| Use case | Dataset |
|----------|---------|
| **Headline paper tables tonight** | **`data/retrace_bench/v1_0/`** (fallback) |
| **Pipeline development / synthetic ablations** | **`data/retrace_bench_smoke/`** or regenerated synthetic splits after gates |
| **Realistic GitHub split** | **Not ready** — pending manual review workflow |
| **Full 3,780 final release** | **Do not publish** until smoke + gold-oracle + validator pass at scale and realistic is manually reviewed |

---

## Files changed (implementation)

- `benchmark/retrace_bench/general_taxonomy.py` — status aliases + `canonical_hidden_gold_fields()`
- `benchmark/retrace_bench/scorers_general.py` — canonical gold reads + legacy fallbacks
- `benchmark/retrace_bench/generation/hard_plus_blueprints.py` — canonical `hidden_gold`, status mapping, answer/rubric alignment
- `benchmark/retrace_bench/generation/adversarial_distractors.py` — per-scenario distractor event IDs
- `benchmark/retrace_bench/generation/release_manifest.py` — evidence count fix
- `benchmark/retrace_bench/generation/github_realistic_blueprints.py` — pending annotation status
- `scripts/generate_retrace_bench_final.py` — realistic/calibration manifest annotation semantics
- `scripts/validate_retrace_bench_dataset.py` — strengthened checks + smoke/packaging modes
- `scripts/check_retrace_bench_gold_oracle.py` — new gold-oracle gate
- `scripts/run_retrace_bench_baseline.py` — mkdir for metrics output path

Generated artifacts (local smoke):

- `data/retrace_bench_smoke/`
- `outputs/retrace_bench_smoke/latest_only_hard.predictions.metrics.json`
- `outputs/retrace_bench_smoke/retrieve_all_hard.predictions.metrics.json`

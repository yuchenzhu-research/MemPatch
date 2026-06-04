# ReTrace-Bench — Repository Cleanup Plan (v1.1 canonical)

Goal: the repository should read as a clean evaluation-benchmark artifact, not a
pile of mixed v1.0 / v1.1 / smoke files. v1.1 is canonical; v1.0 is legacy/pilot.

## Principles

- **GitHub keeps**: code, evaluator, validators, runners, generation/scoring
  scripts, documentation, manifests, checksums, dataset cards, small demo
  examples, and the committed paper-facing split data.
- **Hugging Face hosts** the full dataset for distribution (dataset-license
  hosting). The HF bundle's full `scenarios.jsonl` stays local-only/git-ignored
  except for manifests/checksums/cards.
- **v1.0 → legacy/pilot.** Not deleted; moved under `data_legacy/` and described
  as a legacy appendix, never as the canonical benchmark.
- **v1.1 → canonical.** `data/retrace_bench_v1_1/` is the current benchmark.
- **Never commit**: `.env`, API keys, raw API responses, quota logs, caches,
  huge temporary outputs, or old smoke predictions that nobody needs.
- **Do not delete** useful generation/scoring code. **Do not touch** ReTrace-Learn
  (`src/retrace_learn/`, `src/retracemem/`, `data/retrace_learn/`).

## What changed in this pass (done)

- v1.0 splits moved to `data_legacy/retrace_bench_v1_0/` (via `git mv`).
- Development smoke / hard50 / hard150 / hard500-candidate artifacts moved to
  `data_legacy/retrace_bench_dev_artifacts/`.
- Canonical v1.1 generated deterministically (seed 2027) under
  `data/retrace_bench_v1_1/` with per-split `scenarios.jsonl` + `manifest.json`
  + `README.md`.
- `.gitignore` updated: commit public v1.1 splits + legacy package + HF
  manifests/checksums + gold-free annotation packets; keep full HF `scenarios.jsonl`
  and `private_hidden` local-only.
- Tests repointed to the v1.1 canonical layout; legacy v1.0 is not tested as the
  current canonical dataset.

## Keep on GitHub

```
benchmark/retrace_bench/        scorers, api, taxonomy, public_view, generation
benchmark/examples/             small demo scenarios + demo predictions
scripts/                        validate / gold_oracle / baseline / export+score
                                human annotations / build_hf_release_v1_1 / package(v1.0)
docs/retrace_bench/             validation, baseline, human-validation, hf-release,
                                statistical, cleanup reports
data/retrace_bench_v1_1/        main / hard / realistic / calibration (committed)
data_legacy/retrace_bench_v1_0/ legacy pilot (kept, not canonical)
hf_release/retrace_bench_v1_1/  manifests/checksums/card/license/VERSION (tracked);
                                full scenarios.jsonl local-only
outputs/retrace_bench_v1_1/     gold-oracle + offline-baseline metrics/predictions
annotation_packets/retrace_bench_v1_1/   gold-free public packets + entry sheet
```

## Keep local-only / git-ignored

- `hf_release/retrace_bench_v1_1/**/scenarios.jsonl` (full data → upload to HF).
- `data/retrace_bench_v1_1/private_hidden_200_en/` (private evaluation split).
- caches: `.pycache_compile/`, `.pytest_cache/`, `__pycache__/`, `*.pyc`,
  `.DS_Store`, `local/`, external clones/caches.

## Deferred deletions (after v1.1 HF upload is verified)

These are intentionally **kept for now** and only pruned once the v1.1 HF upload
is confirmed and the legacy v1.0 dataset is archived/tagged on HF:

- Old development smoke outputs under `data_legacy/retrace_bench_dev_artifacts/`.
- Any superseded v1.0 prediction dumps that are not referenced by the paper.

Do not delete these in this pass — they are the only remaining record until the
HF archive/tag exists.

## Explicitly out of scope

- ReTrace-Learn code/data (method track) — untouched.
- Raw API responses / model reruns — none in this pass.
- Publishing to Hugging Face — manual, user-driven (see `v1_1_hf_release_plan.md`).

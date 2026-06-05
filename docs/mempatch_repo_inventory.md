# MemPatch repository inventory

**Generated context:** MemPatch unified paper system. Code paths under `retrace_*` name implementation modules; paper language is benchmark-first (see `mempatch_paper_merge_plan.md`).

## Top-level layout

| Path | Role |
|------|------|
| `/Users/yuchenzhu/Desktop/ReTrace/AGENTS.md` | Agent authority: tracks, DPA/`authorize` boundary, experiment tiers, no-go list |
| `/Users/yuchenzhu/Desktop/ReTrace/README.md` | Human entry: setup, benchmark evaluator CLI, model runner, verification commands |
| `/Users/yuchenzhu/Desktop/ReTrace/RETRACE_BENCH.md` | Benchmark-track scope note (partially stale; see below) |
| `/Users/yuchenzhu/Desktop/ReTrace/LICENSE` | Project license |
| `/Users/yuchenzhu/Desktop/ReTrace/pyproject.toml` | Package `retracemem`; discovers `src/*`, `benchmark/*` |
| `/Users/yuchenzhu/Desktop/ReTrace/benchmark/` | **MemPatch-Bench (evaluation-only)** Python package |
| `/Users/yuchenzhu/Desktop/ReTrace/hf_release/retrace_bench_v1_1/` | HF release metadata/manifests (not full scenario JSONL in git) |
| `/Users/yuchenzhu/Desktop/ReTrace/scripts/` | Public CLI entrypoints for validate / evaluate / run model |
| `/Users/yuchenzhu/Desktop/ReTrace/src/retracemem/` | **DPA / authorization / memory store / multi-agent commit** (MemPatch scaffold kernel) |
| `/Users/yuchenzhu/Desktop/ReTrace/src/retrace_learn/` | **Scenario View Builder + Revision Response Policy** runtime and schemas |
| `/Users/yuchenzhu/Desktop/ReTrace/data/retrace_learn/` | Small method-side JSONL splits + manifest (not full bench) |
| `/Users/yuchenzhu/Desktop/ReTrace/tests/` | Pytest: `authorize` core + benchmark API |
| `/Users/yuchenzhu/Desktop/ReTrace/docs/` | MemPatch planning docs (this file + merge/refactor plans) |
| `/Users/yuchenzhu/Desktop/ReTrace/local/` | **Gitignored** — bench downloads, predictions, results, ad-hoc training/run scripts |
| `/Users/yuchenzhu/Desktop/ReTrace/.venv/` | Local virtualenv (not tracked) |

**Absent but referenced in authority docs:** `experiments/` (STALE/E4 adapters per `AGENTS.md`), `references/`, tracked paper `.tex` drafts, `tests/retrace_bench/` subdirectory.

---

## Classification by MemPatch concern

### Benchmark construction & release packaging

- `/Users/yuchenzhu/Desktop/ReTrace/hf_release/retrace_bench_v1_1/` — VERSION, manifests, checksums, split README, dataset_info (scenarios live on Hugging Face `Sylvan-Vale-Moon/ReTrace-Bench`).
- `/Users/yuchenzhu/Desktop/ReTrace/benchmark/retrace_bench/general_taxonomy.py` — domains, task types, decisions, memory statuses, failure modes, public forbidden terms.
- `/Users/yuchenzhu/Desktop/ReTrace/benchmark/retrace_bench/public_view.py` — gold-stripping public scenario view for model inputs.
- `/Users/yuchenzhu/Desktop/ReTrace/scripts/validate_retrace_bench_dataset.py` — JSONL schema/taxonomy/leakage-oriented validation for bench files.

**Note:** Scenario *generation* pipelines that historically built the 3k/500/200 splits are **not** present in the trimmed repo; construction artifacts are external (HF + git history).

### Benchmark evaluation

- `/Users/yuchenzhu/Desktop/ReTrace/benchmark/retrace_bench/api.py` — `load_scenarios`, `load_predictions`, `evaluate_predictions` (stable external API).
- `/Users/yuchenzhu/Desktop/ReTrace/benchmark/retrace_bench/scorers_general.py` — per-case scoring, headline/auxiliary metrics, aggregation.
- `/Users/yuchenzhu/Desktop/ReTrace/scripts/evaluate_retrace_bench_predictions.py` — CLI wrapper for submissions.
- `/Users/yuchenzhu/Desktop/ReTrace/tests/test_benchmark_api.py` — API/smoke tests.

### MemPatch scaffold / method runtime

| Component | Primary paths |
|-----------|----------------|
| Shared candidate view & method contracts | `src/retracemem/methods/contracts.py` |
| Scenario View Builder | `src/retrace_learn/runtime/graph_extractor.py` (`RuleBasedGraphExtractor`, `LearnedGraphExtractor`) |
| Revision Response Policy | `src/retrace_learn/runtime/learned_proposer.py`, `src/retrace_learn/runtime/dpa_runtime.py` (parse/validate actions) |
| Training signal helpers | `src/retrace_learn/runtime/reward.py`, `path_ranker.py`, `views.py` |
| Method schemas | `src/retrace_learn/schemas.py` |
| Multi-agent submission → proposals | `src/retracemem/multiagent/parser.py`, `contracts.py`, `utils.py` |
| Commit wrappers | `src/retracemem/multiagent/commit.py` — `commit_subagent_submission`, sequence commit → `authorize` |
| Memory substrate | `src/retracemem/memory/belief_store.py`, `episode_ledger.py`, `temporal_validity.py` |
| Canonical runtime types | `src/retracemem/schemas.py` |

### DPA / authorization (deterministic commit)

- **Public kernel:** `src/retracemem/authorization.py` — `authorize(...)`, `EvidenceProposalBatch`, `AuthorizationResult` (sole external entry for deterministic commit).
- **RevisionGate:** `src/retracemem/tms/gate.py` — structural admission of typed edges.
- **DPA algorithm:** `src/retracemem/tms/authorization.py` — `DefeatPathAuthorizationAlgorithm`.
- **Rollback / TMS helpers:** `src/retracemem/tms/rollback.py`.
- **Tests:** `tests/test_authorize_core.py`.

Typed edge vocabulary in code: `SUPERSEDES`, `BLOCKS`, `RELEASES`, `UNCERTAIN`, `REAFFIRMS`, `NO_REVISION` (+ `REQUIRES` dependencies).

### Open-source model adaptation (mostly local / needs_review)

Tracked repo has **hooks** for learned `generate_fn` proposers/extractors but **no** checked-in training scripts under `scripts/`.

Gitignored `/Users/yuchenzhu/Desktop/ReTrace/local/` (inspect locally, do not commit):

- `local/make_retrace_sft_v3.py` — SFT corpus generation (`needs_review` for MemPatch-RSFT naming).
- `local/run_mlx_retrace_struct_predictions.py` — MLX structural predictions on bench.
- `local/run_main96_all_models_parallel.sh` — parallel multi-model bench runs.
- `local/adapters/`, `local/models/`, `local/predictions/`, `local/results/`, `local/logs/` — run artifacts.

Method-side labeled data (small, tracked):

- `data/retrace_learn/v1/boundary_audit/*.jsonl`, `data/retrace_learn/v1/manifest.json`.

### Closed-source Skill.md / procedural policy adaptation

- **Not present in tracked tree.** `AGENTS.md` lists ReTrace-SkillOpt / `memory_policy.md` / Microsoft SkillOpt as **backlog, out of active scope**.
- Planned doc slots (not yet in repo): `docs/skill.md`, `docs/memory_policy.md` (see `mempatch_refactor_plan.md`).

Closed-source baselines today are represented only by:

- `benchmark/retrace_bench/model_runner.py` + `scripts/run_retrace_bench_model.py` (API providers: OpenAI-compatible, Anthropic, etc.) producing **bench prediction JSONL**, not full MemPatch scaffold routing.

---

## Entry-point scripts (tracked)

| Script | Purpose |
|--------|---------|
| `scripts/evaluate_retrace_bench_predictions.py` | Score predictions vs scenarios; write metrics/scored JSONL |
| `scripts/run_retrace_bench_model.py` | Run provider model on bench; write canonical predictions |
| `scripts/validate_retrace_bench_dataset.py` | Validate bench JSONL before release or local checks |

**Python import entrypoints (library):**

- `from benchmark.retrace_bench.api import load_scenarios, evaluate_predictions, ...`
- `from retracemem.authorization import authorize`
- `from retracemem.multiagent.commit import commit_subagent_submission`
- `from retrace_learn.runtime.graph_extractor import RuleBasedGraphExtractor`
- `from retrace_learn.runtime.learned_proposer import LearnedTypedRevisionProposer`

---

## Data generation, experiments, tables/figures

| Activity | Where today | MemPatch note |
|----------|-------------|---------------|
| Bench scenario JSONL | Hugging Face + `local/ReTrace-Bench/` download | Rename to MemPatch-Bench in paper/HF metadata later |
| Prediction generation | `scripts/run_retrace_bench_model.py`, `local/run_*` | Main results tables feed from `evaluate_*` metrics JSON |
| Metrics / tables | `evaluate_retrace_bench_predictions.py` `--print-table`, `--out-metrics` | No LaTeX figure generator in repo |
| SFT / RSFT corpora | `local/make_retrace_sft_v3.py` | `needs_review` — wire to `adaptation/open_source/` in refactor |
| Ablations (gate bypass, structure-only) | `authorize(..., bypass_gate=)`, reward/runtime flags in `retrace_learn` | Document in §7; code scattered in runtime |
| STALE/CUPMem E4 | Was `experiments/` | **Removed** from tree; restore under `mempatch/experiments/` if E4 returns |

---

## Paper files and roles

**No paper source (`.tex`, `.docx`) is tracked** in this repository. README and `RETRACE_BENCH.md` explicitly exclude paper drafts from git.

Roles for writing:

- **Benchmark §4:** `hf_release/retrace_bench_v1_1/README.md`, `general_taxonomy.py`, scorer metric names in `scorers_general.py`.
- **Problem formulation §3:** `retracemem/schemas.py`, `AGENTS.md` vocabulary, `methods/contracts.py`.
- **Scaffold §5:** `authorization.py`, `tms/*`, `multiagent/commit.py`, `retrace_learn/runtime/*`.
- **Experiments §6–7:** README run recipes; historical numbers likely in ignored `local/results/` (`needs_review`).

MemPatch is one unified paper; benchmark response fields are the paper-facing interface.

---

## Duplicated, stale, experimental, or unclear

| Item | Status |
|------|--------|
| `RETRACE_BENCH.md` cites `tests/retrace_bench/` | **Stale** — tests live in `tests/test_*.py` |
| `AGENTS.md` cites `experiments/` for STALE | **Stale** — directory missing post-prune |
| Dual package roots `retracemem` vs `retrace_learn` | **Intentional split** (kernel vs learned stages); consolidate naming under MemPatch scaffold in docs first |
| Two `schemas.py` files | **Overlap risk** — method vs kernel types; mark `needs_review` before merge into one module |
| `benchmark/retrace_bench/model_runner.py` in bench package | **Boundary blur** — runner is evaluation harness, not construction; keep bench method-neutral |
| `local/*` scripts | **Experimental / operational** — not authoritative |
| `realistic` split `synthetic_gold_unreviewed` | **Documented limitation** in HF README |
| Git branch name `benchmark` | **Same commit as main** — merge was no-op at `67f6432` |

---

## Verification surface (current)

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q benchmark scripts src tests
.venv/bin/python -m pytest -q
```

(`experiments/` omitted — directory does not exist.)

Last run on merge check: **9 tests passed**.

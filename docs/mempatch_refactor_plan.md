# MemPatch refactor plan (proposal only)

**Scope:** Planning document only. **Do not** execute large moves in this pass. **Do not** aggressively delete files. Mark uncertain mappings as `needs_review`.

**Goal:** One paper (*MemPatch*), two conceptual layers (**MemPatch-Bench** evaluation vs **MemPatch scaffold** method), aligned directory story:

```
mempatch/
  bench/
    data/
    generators/
    evaluators/
    diagnostics/
  scaffold/
    retrieval/
    proposer/
    dpa/
    commit/
    memory_store/
  adaptation/
    open_source/
    closed_source_skill/
  experiments/
    baselines/
    ablations/
    main_results/
docs/
  skill.md
  memory_policy.md
  mempatch_repo_inventory.md
  mempatch_paper_merge_plan.md
  mempatch_refactor_plan.md
```

---

## Principles

1. **Rename in docs and packaging first**, then optional physical moves with re-export shims.
2. Keep `authorize(...)` as the single public commit API during transition.
3. Preserve `benchmark/retrace_bench` as import path until a deprecation window (`benchmark.retrace_bench` → `mempatch.bench.evaluators` re-exports).
4. Gitignored `local/` remains scratch; promote scripts into `mempatch/adaptation/` only after review.

---

## Phase 0 — Documentation & aliases (low risk)

| Action | Notes |
|--------|-------|
| Keep three docs in `/Users/yuchenzhu/Desktop/ReTrace/docs/` | Done |
| Add stub `docs/skill.md`, `docs/memory_policy.md` | Placeholders for closed-source Skill.md track (`needs_review`) |
| Top-level `MEMPATCH.md` or extend `README.md` | Single human entry under MemPatch benchmark-first language |
| Fix stale pointers | Update `RETRACE_BENCH.md` test path; note `experiments/` absence in `AGENTS.md` when editors touch it |

No file moves.

---

## Phase 1 — Logical grouping without breaking imports

Create package root `mempatch/` as **thin facade** re-exporting existing modules:

| Target | Current source | Move now? |
|--------|----------------|-----------|
| `mempatch/bench/evaluators/` | `benchmark/retrace_bench/api.py`, `scorers_general.py` | **Shim only** |
| `mempatch/bench/evaluators/runner` | `model_runner.py` | **Shim** — consider moving runner to `scripts/` later (`needs_review`) |
| `mempatch/bench/diagnostics/` | `validate_retrace_bench_dataset.py` logic | Extract from script gradually |
| `mempatch/bench/data/` | `hf_release/retrace_bench_v1_1/` | Symlink or copy manifests only; HF remains canonical |
| `mempatch/bench/generators/` | *Missing in repo* | `needs_review` — restore from git history if regeneration needed |
| `mempatch/scaffold/memory_store/` | `src/retracemem/memory/*` | Re-export |
| `mempatch/scaffold/dpa/` | `src/retracemem/tms/authorization.py`, `temporal_validity.py` | Re-export |
| `mempatch/scaffold/commit/` | `src/retracemem/tms/gate.py`, `authorization.py` (public), `multiagent/commit.py` | Re-export |
| `mempatch/scaffold/proposer/` | `src/retrace_learn/runtime/learned_proposer.py`, `dpa_runtime.py` | Re-export |
| `mempatch/scaffold/retrieval/` | `methods/contracts.py`, `retrace_learn/runtime/views.py`, `graph_extractor.py` | Re-export |
| `mempatch/adaptation/open_source/` | `local/make_retrace_sft_v3.py`, `local/run_mlx_*` | **Copy in** after sanitizing secrets/paths |
| `mempatch/adaptation/closed_source_skill/` | Future Skill.md runner | Empty + `needs_review` |
| `mempatch/experiments/baselines/` | `scripts/run_retrace_bench_model.py` wrappers | Script calls into bench runner |
| `mempatch/experiments/main_results/` | `local/run_main96_all_models_parallel.sh` | Document-only pointer until promoted |

Update `pyproject.toml` `include` to add `mempatch*` when facade exists.

---

## Phase 2 — Physical moves (medium risk, batch with tests)

Suggested order (each step: move + compatibility import + pytest):

1. `benchmark/retrace_bench/` → `mempatch/bench/evaluators/` leaving `benchmark/retrace_bench/__init__.py` re-export deprecated.
2. `hf_release/retrace_bench_v1_1/` → `mempatch/bench/data/retrace_bench_v1_1/` or rename to `mempatch_bench_v1_1` on HF sync.
3. Split `src/retracemem/` into `mempatch/scaffold/{memory_store,dpa,commit}` — keep `retracemem` meta-package as alias **one release cycle**.
4. Split `src/retrace_learn/` → `mempatch/scaffold/proposer` + `retrieval` (graph builder).
5. Recreate `mempatch/experiments/stale_adapter/` from git history if E4 returns (`AGENTS.md` isolation rules).

---

## Phase 3 — Naming cleanup (high churn, post-paper submission optional)

| Legacy | MemPatch target |
|--------|-----------------|
| ReTrace-Bench | MemPatch-Bench |
| Former ReTrace-Learn framing | MemPatch scaffold (Scenario View Builder + Revision Response Policy) |
| `retracemem` PyPI name | `mempatch` or keep `retracemem` as kernel subpackage (`needs_review`) |
| HF dataset `Sylvan-Vale-Moon/ReTrace-Bench` | New repo card or revision bump (`needs_review`) |
| `NO_REVISION` | Alias `NO_PATCH` in paper-facing enums only (keep code enum stable until v2) |

---

## Files to treat as `needs_review` before move/delete

- `RETRACE_BENCH.md` (stale test path)
- `AGENTS.md` references to `experiments/`, dual-track paper wording
- Duplicate schemas: `src/retracemem/schemas.py` vs `src/retrace_learn/schemas.py`
- `benchmark/retrace_bench/model_runner.py` (evaluation vs method boundary)
- All of `local/` (operational, may contain paths/API keys)
- `data/retrace_learn/v1/boundary_audit/*.jsonl` (split role vs bench leakage narrative)

---

## Files explicitly **not** targeted for deletion

- Any benchmark scorer logic (`scorers_general.py`) — evaluation contract.
- `tests/test_authorize_core.py` — kernel regression anchor.
- `hf_release/*` manifests — release integrity.
- Git history / backup branch `backup/pre-benchmark-merge` at `67f6432`.

---

## Verification gates per phase

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q mempatch benchmark scripts src tests
.venv/bin/python -m pytest -q
```

Add import smoke test:

```python
from retracemem.authorization import authorize  # legacy
# future: from mempatch.scaffold.commit import authorize
```

---

## Recommended next coding step (after this plan)

**Phase 0 + facade:** Add `mempatch/` re-export package and a single `README` subsection for MemPatch naming **without** moving `benchmark/retrace_bench` yet. Fix `RETRACE_BENCH.md` stale test path in the same PR.

## Recommended next writing step

Draft **§3 Problem Formulation** and **§4 MemPatch-Bench** from `mempatch_paper_merge_plan.md` using HF README + `AGENTS.md`; defer §7 until metrics exported from `local/results/` into a committed `needs_review` table under `docs/` or an ignored artifact path.

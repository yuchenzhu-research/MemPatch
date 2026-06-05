# MemPatch paper merge plan

**Target paper:** *MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents*

**Repo:** `/Users/yuchenzhu/Desktop/ReTrace` (internal code names: ReTrace, ReTrace-Bench, ReTrace-Learn, `retracemem` package).

This map ties **existing files and concepts** to an **8-page** MemPatch narrative. ReTrace appears only where it names code modules or historical artifacts.

---

## Naming translation (use in prose)

| MemPatch (paper) | Current repo / code |
|------------------|---------------------|
| MemPatch | Umbrella system + paper title |
| Rapid Memory Integration (RMI) | Agent must integrate new evidence into usable memory **quickly and correctly** under typed patch semantics |
| MemPatch-Bench | `benchmark/retrace_bench/`, HF `ReTrace-Bench`, `hf_release/retrace_bench_v1_1/` |
| MemPatch scaffold | Graph extraction + typed patch proposal + **DPA** + deterministic commit + audit trace (`src/retrace_learn/runtime/*` + `src/retracemem/*`) |
| Patch action `a` | `EvidenceEdge` / `RevisionAction`: SUPERSEDE (code: `SUPERSEDES`), BLOCK, RELEASE, REAFFIRM, UNCERTAIN, NO_PATCH (code: `NO_REVISION`) |
| Memory `M → M'` | Shared basis after `authorize(...)` / `commit_subagent_submission` — eligibility change, append-only evidence graph |
| DPA | `DefeatPathAuthorizationAlgorithm` in `src/retracemem/tms/authorization.py` |
| Baseline API proposer | `benchmark/retrace_bench/model_runner.py` (predicts bench **response** fields, not full scaffold) |
| Learned proposer / graph builder | `LearnedTypedRevisionProposer`, `LearnedGraphExtractor` |
| Oracle / teacher | `RuleBasedGraphExtractor`, `ScriptedProposer`, E0 replay in tests |

---

## Section 1 — Introduction

**Sources:** `hf_release/retrace_bench_v1_1/README.md` (motivation: stale/out-of-scope beliefs), `AGENTS.md` one-sentence alignment (evidence-preserving typed defeat paths).

**Content to write:**

- **Motivation:** LLM agents accumulate memory; new evidence should patch beliefs without silent staleness or ungrounded updates (RMI).
- **Running example:** One MemPatch-Bench episode — public view only in figure; show gold patch type in caption (from `hidden_gold`, not model input).
- **RMI definition:** Given memory state, new evidence event `e`, produce admissible typed patch actions and a committed basis `M'` such that downstream Q&A uses authorized beliefs only.
- **Contributions (draft bullets tied to repo):**
  1. MemPatch-Bench — 3780 public scenarios, multi-metric revision evaluation (`scorers_general.py` headline metrics).
  2. MemPatch scaffold — typed proposals + deterministic DPA commit (`authorize`).
  3. Learning recipe — Graph Builder + Proposal Policy with DPA-guided rewards (`retrace_learn/runtime/reward.py`).
  4. Empirical study — API baselines via `run_retrace_bench_model.py` + learned/open-weight paths (`local/` runs, `needs_review` for numbers).

**Drop from old dual-paper framing:** Separate “ReTrace-Bench paper” vs “ReTrace-Learn paper” — one contribution list.

---

## Section 2 — Related Work

**Sources:** No `references/` registry in repo — rebuild bib from prior drafts in git history or external notes (`needs_review`).

**Buckets to cover (map to bench taxonomy):**

- Agent memory / long-horizon chat memory (`general_taxonomy.py` domains).
- Memory editing & knowledge update (contrast: MemPatch **typed patch + DPA**, not free-text overwrite).
- Benchmarks for forgetting/conflict (position MemPatch-Bench as **revision reliability** with evidence paths + downstream probes).
- Tool/skill-based agents (pointer only — Skill.md adaptation is future/`docs/skill.md`).

---

## Section 3 — Problem Formulation

**Primary sources:** `AGENTS.md` (canonical edges), `src/retracemem/schemas.py`, `src/retracemem/methods/contracts.py` (`SharedCandidateView`).

**Notation map:**

| Symbol | Code anchor |
|--------|-------------|
| Memory `M` | Candidate beliefs/conditions/evidence in `SharedCandidateView` + store state |
| Evidence `e` | New `EvidenceNode` / submission `new_evidence_id` |
| Affected memories | Target belief/condition ids in proposals |
| Patch action `a` | Typed edge in canonical vocabulary |
| `M → M'` | Append-only graph update + DPA statuses → authorized basis |

**Action table (paper ↔ code):**

| Paper | Code enum | Effect (short) |
|-------|-----------|----------------|
| SUPERSEDE / UPDATE | `SUPERSEDES` | New belief supersedes prior |
| BLOCK | `BLOCKS` | Evidence blocks condition → prerequisite block |
| RELEASE | `RELEASES` | Clears blocker; eligibility not truth |
| REAFFIRM | `REAFFIRMS` | Supports belief without supersession |
| UNCERTAIN | `UNCERTAIN` | Unresolved usability |
| NO_PATCH | `NO_REVISION` | No graph patch; evidence still logged |

**DPA precedence (verbatim from code/docs):** `SUPERSEDES > PREREQUISITE_BLOCK > UNRESOLVED_UNCERTAIN > AUTHORIZED`.

**Figure suggestion:** Pipeline diagram: `view, e → propose(a) → RevisionGate → DPA → M'`.

---

## Section 4 — MemPatch-Bench

**Primary sources:** `hf_release/retrace_bench_v1_1/README.md`, `benchmark/retrace_bench/general_taxonomy.py`, `public_view.py`, `validate_retrace_bench_dataset.py`.

**Subsections:**

1. **Episodes** — JSONL scenarios; splits `main` (3000), `hard` (500), `realistic` (200, unreviewed gold), `calibration` (80 smoke).
2. **Task categories** — `TASK_TYPES`, `DOMAINS`, `DIFFICULTIES`, `FAILURE_MODES` from taxonomy module.
3. **Gold labels** — `hidden_gold` block (scorer-only); public input via official public view.
4. **Evidence path** — `evidence_event_ids`, minimal-evidence constraints (`minimal_evidence_exact_match` metric).
5. **Downstream Q&A** — Scenario `response` / decision fields scored by `scorers_general.py` (`decision_macro_f1`, `joint_revision_success`, etc.).

**Limitation paragraph (required):** `realistic` split disclaimer from HF README; calibration not for model selection.

**Rename in text:** MemPatch-Bench; keep HF dataset rename as follow-up release note.

---

## Section 5 — MemPatch Scaffold

**Primary sources:** `src/retrace_learn/runtime/graph_extractor.py`, `learned_proposer.py`, `src/retracemem/authorization.py`, `multiagent/commit.py`, `tms/gate.py`, `tms/authorization.py`.

**Subsections mapped to implementation:**

| Scaffold stage | Repo component |
|----------------|----------------|
| Retrieval / candidate view | `SharedCandidateView`, `views.py` (bounded context) |
| Typed patch proposal | `LearnedTypedRevisionProposer` / API baseline (bench runner is **not** full scaffold — clarify in text) |
| DPA | `DefeatPathAuthorizationAlgorithm` |
| Commit | `authorize`, `commit_subagent_submission` |
| Audit trail | `AuthorizationResult.trace`, gate decisions in trace dict |

**Multi-agent story (optional subsection):** Subagents submit evidence-bearing updates; shared basis updated only through kernel — aligns with MemPatch RMI in teams.

**Internal name footnote:** “ReTrace-Engine” = Parser + RevisionGate + DPA + audit inside commit path.

---

## Section 6 — Experimental Setup

**Models & baselines (from repo capabilities):**

| Baseline | Implementation |
|----------|------------------|
| API end-task predictor | `run_retrace_bench_model.py` + provider list in `model_runner.py` |
| Direct status judge | Described in `AGENTS.md` as DirectJudge-API — **no dedicated script in trimmed repo** (`needs_review` / reproduce from history) |
| Prompted typed proposer | Stage A in AGENTS — **needs_review** for current script |
| MemPatch scaffold + learned policy | `retrace_learn` runtime + local MLX/SFT scripts |
| Oracle / E0 | `tests/test_authorize_core.py`, rule-based extractors |

**Metrics:** Import from `benchmark/retrace_bench/api.py` — `HEADLINE_METRICS`, `AUXILIARY_METRICS` (table in appendix).

**Data splits for training vs eval:** `data/retrace_learn/v1/manifest.json` for method splits; bench splits eval-only per HF README.

---

## Section 7 — Main Results and Ablations

**What repo supports today:**

- Full bench evaluation pipeline for any model producing canonical predictions JSONL.
- Kernel ablations: `bypass_gate` on `authorize` (RevisionGate off).
- Structure-only / partial prediction tolerance in scorer (recent commits on `benchmark` branch history).

**What needs re-hydration (`needs_review`):**

- LaTeX tables from `local/results/*.metrics.json`.
- Learned vs API comparisons on `main` split.
- Ablations: no DPA, no gate, graph builder error propagation — wire to existing flags + new experiment configs.

**Suggested table rows:** decision F1, memory state acc, evidence F1, joint revision success, format failure rate — all defined in scorer.

---

## Section 8 — Analysis, Limitations, Conclusion

**Analysis angles supported by bench taxonomy:**

- Per-`FAILURE_MODE` breakdown (scorer supports failure diagnosis accuracy).
- Hard split L3/L4 behavior.
- Realistic split caveat.

**Limitations (repo-specific):**

- No tracked STALE bridge (`experiments/` removed).
- Realistic gold unreviewed.
- Training on bench-derived labels for Learn track — disclose per `AGENTS.md` / data manifest.
- Closed-source Skill.md policy path not evaluated in tree.

**Conclusion:** MemPatch unifies **measurement** (MemPatch-Bench) and **mechanism** (scaffold + learning) for RMI.

---

## File-level checklist for co-authors

| Paper section | Must-read paths |
|---------------|-----------------|
| §3 | `AGENTS.md`, `src/retracemem/schemas.py`, `authorization.py` |
| §4 | `hf_release/retrace_bench_v1_1/README.md`, `scorers_general.py`, `public_view.py` |
| §5 | `retrace_learn/runtime/*`, `multiagent/commit.py`, `tms/*` |
| §6–7 | `README.md`, `scripts/*`, `local/results/` (local) |
| Appendix metrics | `benchmark/retrace_bench/api.py`, `general_taxonomy.py` |

---

## Writing order (recommended)

1. §3 + §5 from code (stable APIs).
2. §4 from HF README + taxonomy (minimal paraphrase).
3. §6–7 after pulling metrics from `local/results/`.
4. §1–2 last (contributions and related work once numbers fixed).

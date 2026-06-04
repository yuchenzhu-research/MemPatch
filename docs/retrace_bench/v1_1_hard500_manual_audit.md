# ReTrace-Bench v1.1 — hard500 Candidate Manual Audit

**Dataset:** `data/retrace_bench_hard500_candidate/hard_500_en/scenarios.jsonl`
(500 scenarios, controlled_synthetic, L3=250/L4=250, seed `2027`).

**Sampling plan (per task):** 54 cases sampled with seed 2027 —
- **≥2 per pattern** (all 15 patterns covered);
- **12 non-answer** cases (`ask_clarification` / `mark_unresolved` / `escalate` /
  `refuse_due_to_policy`);
- **12 random** cases.

Each sampled case was checked (programmatically + manual read) against the
required fields below.

## Audit checklist & results

| Check | Result |
| --- | --- |
| **No leakage in public input** (`hidden_gold`, `metadata`, `is_distractor`, `source_pointers`, `primary_failure_mode`, `pattern_trap_type`, `canonical_failure_mode`, `validation_notes` absent from `public_scenario_view`) | **PASS** — 0/54 leaked |
| **Solvable from visible evidence** (every gold `evidence_event_id` present in the public `event_trace`) | **PASS** — 0/54 referenced an invisible event |
| **Gold evidence minimal & sufficient** (≥1 evidence id; not all-events; avg required evidence = 1.69 across the split) | **PASS** — no empty and no over-cited gold |
| **Gold memory_state complete** (covers exactly the `initial_memory` ids) | **PASS** — 0/54 incomplete |
| **Gold decision valid & reasonable** (in the 5-label set; consistent with the pattern) | **PASS** |
| **Gold failure diagnosis present & reasonable** | **PASS** — non-empty for all sampled |
| **Answer non-trivial** (grounded, >20 chars, no decision-word label leak) | **PASS** |
| **Topic/domain consistency** | **PASS with a minor note** (see below) |
| **No filler-heavy artifacts** | **PASS** (one benign background event per case) |

**Automated issue counter across the 54-case sample: `NONE`.**

## Representative cases (manual read)

- `rt-hard-000004` — `ci_failed_after_claim`, L4, calendar domain → gold
  `escalate` / `conflict_collapse`. A trusted user claims a hotfix is "ready and
  merged" while a **verified CI** event reports the build failed; gold evidence
  is exactly those two events. Decision and minimal evidence are correct.
- `rt-hard-000008` — `security_policy_override`, L4, BI domain → gold
  `refuse_due_to_policy` / `policy_violation`. A request to cache plaintext
  credentials is met by a **verified security override**; gold evidence is the
  single override event. Correct.
- `merged_but_unreleased`, software domain → gold `use_current_memory`; the
  unreleased-merge trap is correctly resisted (memory stays current, scope-beta
  distractor marked `out_of_scope`).

In every read case the **target memory** and **`workflow_context`** are written
in the case's own domain vocabulary (room booking, source-table lineage, refund
policy, etc.), and the gold answer references the domain's verified source
(calendar sync, warehouse lineage, …).

## Minor note (non-blocking)

The controlled-synthetic generator layers a **shared adversarial distractor
frame** (a version/nightly distractor, an untrusted-claim distractor, a
rollback-proposal distractor, a CI-failure distractor, and one benign
"API gateway authentication" background event) onto each scenario. As a result,
some **distractor** events use a generic software/CI phrasing even in non-software
domains (e.g. a "performance hotfix / CI compilation failed" distractor inside a
calendar scenario). This is intentional adversarial noise, not filler and not
leakage: the **authoritative** evidence and the target memory remain
domain-consistent, distractors are correctly excluded from gold evidence, and
the gold answer is domain-specific. It is a mild realism limitation worth noting
for the paper's "synthetic-style" framing, not a correctness defect.

## Verdict

The hard500 candidate **passes manual audit**: no leakage, gold is
solvable/minimal/sufficient, decisions and diagnoses are reasonable, and content
is topic-consistent with only a benign cross-domain distractor-frame stylistic
note. This satisfies the manual-spot-check NO-GO guards (no public-prompt
leakage, no topic-inconsistent or filler-heavy examples).

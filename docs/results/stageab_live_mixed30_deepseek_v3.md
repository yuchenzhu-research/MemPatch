# Live Stage A vs Stage B — DeepSeek-V3, 30 mixed dev cases

Curated record of a **live** Stage A vs Stage B run on a 30-case mixed slice of
`dev_expansion`. Raw artifacts live under `outputs/runs/<run_id>/` (gitignored
per `AGENTS.md`); this note is the durable, reviewable summary. Companion to the
8-case smoke in `stageab_live_smoke_deepseek_v3.md`.

## Run manifest (no secrets)

| field | value |
| --- | --- |
| git commit | `076f3f3` |
| provider / mode | `siliconflow` / `openai-chat` |
| model | `deepseek-ai/DeepSeek-V3` |
| temperature | 0.0 |
| constrained | true |
| cache | true |
| data split | `dev_expansion` |
| max cases | 30 (→ 55 beliefs) |
| prompt_version | `0abff0a4…` (sha256) |
| api key env | `SILICONFLOW_API_KEY` (present: true) |
| live API run | **true** |

## Slice composition (the "mixed" part)

The first 30 `dev_expansion` cases span **6 phenomenon types** (all in the
software-engineering domain; the research-workflow domain begins at case 45 and
is not included):

`direct_supersession` ×5, `stale_propagation` ×5, `scope_expansion` ×5,
`cross_agent_conflict` ×5, `temporary_blocker_recovery` ×5, `duplicate_evidence` ×5.

## Headline result

| stage | final-status accuracy (55 beliefs) |
| --- | --- |
| **Stage A** (typed actions → RevisionGate → DPA) | **52/55 = 0.945** |
| **Stage B** (DirectJudge) | **15/55 = 0.273** |

Stage B strict == canonicalized == 0.273 → genuine judgment gap, not a
canonicalization artifact. **Stage A wrong while Stage B correct: 0.**

## Stage A failure mode (finally non-empty)

All **3** Stage A errors are the same category — `uncertainty_collapse` — and all
on `cross_agent_conflict`:

```
gold UNRESOLVED  ->  A AUTHORIZED   (proposer emitted REAFFIRMS, gate admitted ok)
```

The zero-shot proposer treated a two-agent conflict as a confirmation
(`REAFFIRMS`) instead of flagging it (`UNCERTAIN`); DPA then deterministically
authorized it. This is a **proposer/prompt weakness, not a gate/DPA bug** (the
gate correctly admitted a well-formed REAFFIRMS; the model simply chose the
wrong action). Suggested fix recorded in the CSV: *"Proposer collapsed UNCERTAIN
to a hard status; add UNCERTAIN affordance/exemplars."* — i.e. exactly the kind
of thing Stage C API-ICL is meant to target.

## Stage B failure modes (40 wrong beliefs)

| gold → Stage B final | count | reading |
| --- | --- | --- |
| AUTHORIZED → UNCERTAIN | 15 | under-update / omitted-or-hedged verdicts |
| SUPERSEDED → USABLE | 10 | stale propagation (never retracts old belief) |
| UNRESOLVED → NOT_USABLE | 5 | over-commits a conflicted belief |
| BLOCKED → UNCERTAIN | 5 | fails to apply prerequisite block |
| AUTHORIZED → NOT_USABLE | 5 | over-update (rejects a usable belief) |

The direct judge has no structural mechanism to (a) retract superseded beliefs,
(b) propagate prerequisite blocks, or (c) reliably authorize replacements — which
is the whole point of the typed-action + DPA decomposition in Stage A.

## Honest takeaways

- On this mixed slice Stage A is **far** better than Stage B (0.945 vs 0.273) and
  is never worse on any single belief.
- Stage A is **not** perfect: cross-agent conflict is its weak spot
  (`uncertainty_collapse`), and it is a proposer-side issue.
- Still a dev slice (30/70, single domain) — directional, not a headline metric.
  The natural follow-ups: full dev70 incl. research-workflow domain, and a Stage C
  API-ICL run targeting the cross-agent-conflict UNCERTAIN gap (requires
  human-approved exemplars — currently fail-closed).

## Reproduce

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S); COMMIT=$(git rev-parse --short HEAD)
env PYTHONPATH=. SILICONFLOW_API_KEY="$SILICONFLOW_API_KEY" .venv/bin/python \
  experiments/multiagent/run_stageab_api_eval.py \
  --live --provider siliconflow --model deepseek-ai/DeepSeek-V3 \
  --constrained --max-cases 30 \
  --output-dir outputs/runs/stageab_live_mixed30_deepseek_v3_${COMMIT}_${RUN_ID}

PYTHONPATH=. .venv/bin/python scripts/build_failure_analysis.py \
  --run-dir outputs/runs/stageab_live_mixed30_deepseek_v3_${COMMIT}_${RUN_ID}
```

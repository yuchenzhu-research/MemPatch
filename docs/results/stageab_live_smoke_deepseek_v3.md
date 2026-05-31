# Live Stage A vs Stage B smoke — DeepSeek-V3 (8 cases)

Curated record of a **live** Stage A vs Stage B smoke run. Raw run artifacts
(`failure_analysis.csv/.md/manifest.json`, parsed/raw/trace JSONL, cache) live
under `outputs/runs/<run_id>/` and are gitignored per `AGENTS.md`; this note is
the durable, reviewable summary.

## Run manifest (no secrets)

| field | value |
| --- | --- |
| git commit | `cb0b291` |
| provider / mode | `siliconflow` / `openai-chat` |
| model | `deepseek-ai/DeepSeek-V3` |
| temperature | 0.0 |
| constrained | true |
| cache | true |
| data split | `dev_expansion` |
| max cases | 8 (→ 19 beliefs) |
| prompt_version | `0abff0a4…` (sha256) |
| api key env | `SILICONFLOW_API_KEY` (present: true) |
| live API run | **true** |

The manifest records only the env-var *name* and a presence bool; the key value
is never read, printed, or stored.

## Headline result

| stage | final-status accuracy (19 beliefs) |
| --- | --- |
| **Stage A** (ReTrace-API-ZeroShot: typed actions → RevisionGate → DPA) | **19/19 = 1.00** |
| **Stage B** (DirectJudge-API: direct status) | **0/19 = 0.00** |

Stage B strict and canonicalized accuracy are **both 0.00** — so this is a
genuine judgment gap, not a metric-canonicalization artifact.

> Honesty note: the original task framing asked "where is Stage A *worse* than
> Stage B." On this slice the opposite holds — Stage A is strictly better, and
> there are **0** beliefs where Stage A is wrong while Stage B is correct. The
> Stage-A-centric `failure_category` column is therefore all `none`.

## Why Stage B fails here (the real divergence)

Every episode in this slice is a cumulative **supersession / stale-propagation**
case (first 8 `dev_expansion` cases, software-engineering domain). The
divergence is fully consistent across all 8:

1. **Stale propagation / over-update (8/19):** for the original belief that the
   second submission supersedes, the direct judge keeps it `USABLE`
   (gold `SUPERSEDED`). It never retracts the stale belief.
   - `gold SUPERSEDED → A SUPERSEDED ✓ , B USABLE ✗`
2. **Omitted verdicts → uncertainty (11/19):** in all 8 episodes the DirectJudge
   response for the *second* submission **omitted verdicts** for the newly
   introduced replacement / dependent beliefs (parse_error: "DirectJudge
   response omitted verdicts for belief(s) …"). With no usable verdict those
   beliefs resolve to `UNCERTAIN` (gold `AUTHORIZED` or `BLOCKED`).
   - `gold AUTHORIZED → A AUTHORIZED ✓ , B UNCERTAIN ✗`
   - `gold BLOCKED    → A BLOCKED ✓    , B UNCERTAIN ✗`

Stage A handles both cleanly: the model proposes typed `SUPERSEDES` /
`REAFFIRMS` actions, RevisionGate admits them (`admit(ok)`), and the
deterministic DPA both supersedes the old belief and authorizes the replacement.
Stage B's `valid_output_rate` is 0.50 precisely because it dropped beliefs on
the second submission of every episode.

## Representative rows (full CSV: `outputs/runs/<run_id>/failure_analysis.csv`)

| case | belief role | gold | Stage A | Stage B (raw → final) | A✓ | B✓ |
| --- | --- | --- | --- | --- | --- | --- |
| case_000 | original (superseded) | SUPERSEDED | SUPERSEDED | USABLE → USABLE | ✓ | ✗ |
| case_000 | replacement | AUTHORIZED | AUTHORIZED | MISSING → UNCERTAIN | ✓ | ✗ |
| case_006 | blocked condition belief | BLOCKED | BLOCKED | MISSING → UNCERTAIN | ✓ | ✗ |

## Caveats

- **8 cases is a smoke, not an evaluation.** This slice is homogeneous
  (supersession/stale only, one domain) and was chosen for cheap signal — it is
  not representative of dev70 and must not be reported as a headline metric.
- The interesting next step is a larger, **mixed** dev slice to find cases where
  the direct judge's canonicalization actually *helps* it
  (`Stage_B_canonicalization_advantage`) and any cases where Stage A's proposer
  is the weak link.

## Reproduce

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S); COMMIT=$(git rev-parse --short HEAD)
env PYTHONPATH=. SILICONFLOW_API_KEY="$SILICONFLOW_API_KEY" .venv/bin/python \
  experiments/multiagent/run_stageab_api_eval.py \
  --live --provider siliconflow --model deepseek-ai/DeepSeek-V3 \
  --constrained --max-cases 8 \
  --output-dir outputs/runs/stageab_live_smoke_deepseek_v3_${COMMIT}_${RUN_ID}

PYTHONPATH=. .venv/bin/python scripts/build_failure_analysis.py \
  --run-dir outputs/runs/stageab_live_smoke_deepseek_v3_${COMMIT}_${RUN_ID}
```

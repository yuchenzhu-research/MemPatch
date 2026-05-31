# Evaluation guide (Stage A / B / C)

Stages are **method variants over one shared pipeline**, not a linear runtime
chain. Stage A and Stage C are typed-action proposer families that feed the same
`commit → DPA → authorize(...)` path; Stage B is a direct-judge baseline.

| Stage | Name | Pipeline |
| --- | --- | --- |
| A | `ReTrace-API-ZeroShot` / `ReTrace-Prompt` | typed proposer → RevisionGate → DPA |
| B | `DirectJudge-API` | direct usability verdict; **no** typed actions / gate / DPA |
| C | `ReTrace-AdaptiveProposer` | adaptive typed proposer → same commit / DPA path |

## Offline smoke (no API key)

```bash
# Compile + full offline suite
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests experiments
python3 -m pytest

# Stage A vs Stage B, mock replay (8 cases), constrained proposer
python3 scripts/evaluate.py stage-ab --mock --max-cases 8 --constrained \
    --output-dir outputs/runs/smoke_stageab

# Stage C, offline mock generations
python3 scripts/evaluate.py stage-c --smoke --max-cases 4 \
    --output-dir outputs/runs/smoke_stagec
```

Each run writes a structured directory: `metrics.json`, `manifest.json`
(commit/provider/model/temperature/prompt-hash/dataset provenance),
`failure_breakdown.csv`, and `stage_*_raw|parsed.jsonl` + `dpa_traces.jsonl`.

## Reading the Stage A vs Stage B comparison

`failure_breakdown.csv` is the per-case comparison table:

```text
episode_id | failure_type | accuracy_a | accuracy_b_strict | accuracy_b_canonicalized
           | has_parse_error_a/b | has_grounding_error_a/b
```

Stage B reports **strict** and **canonicalized** accuracy separately
(`canonicalization_rate` shows how often belief-id canonicalization was applied):
canonicalization is a metric-side convenience for Stage B verdicts and is kept
distinct from the strict score so it cannot inflate the headline comparison.

### Where to look when Stage A trails Stage B

The typed-decomposition metrics localize *why* Stage A diverges, separately from
the final-status score:

| Symptom (Stage A metric) | Likely cause |
| --- | --- |
| `valid_json` / `parser_error_rate` | malformed proposer output |
| `action_type_match` low | wrong action chosen (e.g. missed SUPERSEDES) |
| `target_grounding` low | invalid target id (gate would reject) |
| `evidence_grounding` low | action not grounded in the cited new evidence |
| `false_no_revision_rate` high | `NO_REVISION` overuse (under-update) |
| `over_update_rate` | unwarranted revisions |

> **Offline note.** Mock/replay modes are deterministic oracle replays, so Stage
> A and Stage B both reach 1.0 final-status accuracy on the smoke cases and show
> no divergence. A genuine "Stage A worse than Stage B" analysis requires a
> **live** model run (`--live` with a provider key); the metrics and
> `failure_breakdown.csv` above are the tools used to attribute those live
> divergences. Results are never fabricated for runs that were not executed.

## Live runs

See [`api_providers.md`](api_providers.md). Use `temperature 0`, a small
`--max-cases`, and a timestamped `--output-dir`. The manifest records the git
commit, provider, model, and prompt-template hash for reproducibility.

## Stage C status (honest)

Stage C is executable **offline** today: `--mock` (deterministic `NO_REVISION`
generations) and `--generations-dir` (replay decoded adapter/SFT generations
through the identical commit/DPA path). The API-ICL proposer
(`ClosedAPIICLProposer`) exists with exemplar retrieval but is **not** wired to a
turnkey live CLI and is intentionally fail-closed without human-approved
exemplars (per `AGENTS.md`). LoRA/SFT is offline data-prep plus an example
config. Stage C live/API-ICL is therefore **not** claimed as done.

# Experiment Protocol

## Method variants (the canonical comparison)

Stage A, Stage B, and Stage C are **method variants over the same shared
pipeline**, evaluated on the **same fixed-candidate cases** for fairness.

- **Stage A — `ReTrace-API-ZeroShot`**: zero-shot prompted typed proposer →
  `RevisionGate` → DPA.
- **Stage B — `DirectJudge-API`**: predicts a final usability verdict directly;
  no typed actions, gate, or DPA. Baseline only.
- **Stage C — `ReTrace-AdaptiveProposer`**: adaptive typed-proposer family
  (API-ZeroShot, API-ICL, Hosted-FT, Open LoRA-SFT) → the *same* commit / DPA
  path as Stage A.

Stage A and Stage B are run **jointly** by the A-vs-B runner so both are scored
on identical inputs; `scripts/evaluate.py stage-a` and `stage-b` are aliases
into that joint runner.

## Inputs and the leakage boundary

A proposer policy consumes only **method-visible** structure:

- a prior shared-memory context / bounded candidate view,
- an evidence-bearing subagent submission,
- candidate beliefs and candidate replacement beliefs,
- conditions and **pre-existing** `REQUIRES` anchors.

A live policy may see conditions and pre-existing `REQUIRES` anchors (they are
method-visible candidate structure) but may **never** see typed gold revision
targets or evaluator final statuses. Every live typed action must explicitly
cite the visible new evidence grounding the proposed revision.

## Metrics

Computed by `retracemem.evaluation.multiagent.metrics` (pure functions):

- final-status accuracy (DPA status vs. gold snapshot),
- typed-action metrics (exact action match, action-type match) for Stage A/C,
- evidence-grounding correctness,
- failure-mode breakdown (parse failure, grounding error, etc.).

## Running

```bash
# Stage A/B (offline or live)
python3 scripts/evaluate.py stage-a --mock
python3 scripts/evaluate.py stage-a --live --provider siliconflow --model deepseek-ai/DeepSeek-V3 --constrained

# Stage C (replay decoded adapter/SFT generations through commit/DPA)
python3 scripts/evaluate.py stage-c --generations-dir path/to/generations --policy-variant lora_sft

# Stage C dataset export (offline)
python3 scripts/export_stagec_data.py
```

## Stage C training / promotion gate

- Only **human-approved reviewed examples** may be used for live smoke or
  training export. No development candidate is promoted until a human review
  decision is recorded against an immutable review-pack manifest hash.
- Stage C **training** code lives in `experiments/multiagent/local_training/`
  and is out of the evaluation path. The final commit always remains
  deterministic and API-free.

## Paper experiment hierarchy (per `AGENTS.md`)

- **E0** Oracle/Replay kernel validation (mechanism verification).
- **E1** Fixed-candidate revision evaluation (primary controlled comparison —
  Stage A vs Stage B vs Stage C on identical candidate contexts).
- **E2** Stage C training and model-driven proposal evaluation.
- **E3** Closed-loop multi-agent workflow (shared memory affects downstream).
- **E4** STALE/CUPMem external validation / compatibility analysis.

Historical implementations of the action-ablation and composition studies, and
the E4 STALE/CUPMem external validation, are archived under
`experiments/archive/`; if needed for final paper numbers they should be
reimplemented through the shared pipeline rather than revived as-is.

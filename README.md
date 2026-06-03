# ReTrace

**ReTrace** is the umbrella project for reliable shared-memory revision in
multi-agent/agentic workflows. It is governed as **two active research tracks**
(see [`docs/project_governance.md`](docs/project_governance.md)):

1. **ReTrace-Bench** — the benchmark track. An evaluation-only benchmark for
   agent memory revision reliability; it does **not** depend on any training
   method. Owns benchmark data, schema, scoring, baselines, the four v1.0
   evaluation splits, and leakage checks (`benchmark/retrace_bench/`,
   `data/retrace_bench/`, `docs/retrace_bench/`). Clean training and validation
   datasets for the method track reside under `data/retrace_learn/v1_0/`.
2. **ReTrace-Learn** — the method track. A trainable framework that turns
   shared-memory revision authorization into a verifiable learning problem:
   models learn to extract graphs and propose structured revision actions, and a
   deterministic **Authorization Court** (implemented by **ReTrace-Engine**,
   `authorize(...)`) provides execution, evaluation, audit traces, and training
   feedback (`src/retrace_learn/`, `src/retracemem/`).

> **ReTrace-Engine** is the implementation name for the deterministic
> Authorization Court **inside ReTrace-Learn**. It is not a standalone paper or a
> separate top-level track.

| Track / paper | Role | Start here |
| --- | --- | --- |
| ReTrace-Bench | Benchmark/resource paper for agent memory revision reliability. | [`RETRACE_BENCH.md`](RETRACE_BENCH.md), [`benchmark/README.md`](benchmark/README.md), [`papers/retrace_bench/`](papers/retrace_bench/) |
| ReTrace-Learn | Method paper for trainable typed revision proposal with deterministic authorization. | [`RETRACE_LEARN.md`](RETRACE_LEARN.md), [`docs/retrace_learn_pipeline.md`](docs/retrace_learn_pipeline.md), [`papers/retrace_learn/`](papers/retrace_learn/) |

The rest of this README describes the **ReTrace-Learn** method track; for the
benchmark track see [`docs/retrace_bench/`](docs/retrace_bench/).
For a conceptual project map, layout, governance, and dataset definitions, see:
- [`PROJECT_MAP.md`](PROJECT_MAP.md) — Conceptual map of ReTrace-Bench and ReTrace-Learn.
- [`docs/project_governance.md`](docs/project_governance.md) — Two-track project governance authoritative document.
- [`data/README.md`](data/README.md) — Dataset definitions and directory structure details.
- [`src/README.md`](src/README.md) — Source code directory structure details.
- [`benchmark/README.md`](benchmark/README.md) — Benchmark evaluation code directory details.
- [`docs/repo_layout.md`](docs/repo_layout.md) — Repository layout and artifact placement guide.

## Central Research Claim (ReTrace-Learn)

> **ReTrace-Learn** turns shared-memory revision authorization into a verifiable learning problem: models learn to propose structured revision actions, while the deterministic Authorization Court (ReTrace-Engine) provides execution, evaluation, audit traces, and training feedback.

Rather than relying purely on hand-written rules or black-box LLM status predictions, the contribution consists of:
```text
benchmark/task definition
+ learned graph extraction (raw content -> structured graph JSON)
+ learned typed revision proposal (graph JSON + snapshot -> action JSON)
+ deterministic authorization engine (Parser + RevisionGate + Defeat-Path Authorization)
+ DPA-in-the-loop training signal (reward / SFT / DPO feedback)
+ strong baselines and external validation
```

---

## Core Pipeline

The overall architecture routes raw content through learned and deterministic modules:

```text
Raw multi-subagent content / dialogue
    → ReTrace-Learn Graph Extractor         (learned)
    → structured evidence / belief / condition / dependency graph JSON
    → ReTrace-Learn Typed Revision Proposer  (learned)
    → typed revision action JSON
    → Authorization Court — ReTrace-Engine    (deterministic)
        → Parser
        → RevisionGate
        → Defeat-Path Authorization (DPA)
    → final memory statuses + audit trace
```

The Authorization Court (ReTrace-Engine) executes this pipeline via a single public entrypoint:
```python
authorize(view, proposal_batches, *, audit_metadata=None) -> AuthorizationResult
```
* Neither **Defeat-Path Authorization (DPA)** nor `RevisionGate` is called directly by external clients; all admission, routing, and defeat-path computations happen inside `authorize`.
* `commit_subagent_submission(...)` / `commit_submission_sequence(...)` are multi-agent integration wrappers.

---

## Action Vocabulary vs. Final DPA Status

A proposer emits **typed revision actions** over candidate structures using a minimal, expressive action set. The action object is **closed-world** — `action_type` from the canonical enum and every id slot drawn from the candidate graph / visible evidence, with no open-ended `scope` field. Every action (including `NO_REVISION`) must cite at least one grounding `evidence_id`:

```json
{
  "action_type": "SUPERSEDES",
  "target_belief_id": "...",
  "target_condition_id": null,
  "replacement_belief_id": "...",
  "evidence_ids": ["..."],
  "rationale": "short optional explanation"
}
```

| Action | Meaning |
| --- | --- |
| `SUPERSEDES` | New evidence replaces a prior belief (requires `target_belief_id` + `replacement_belief_id`) |
| `BLOCKS` | New evidence blocks a prerequisite condition (requires `target_condition_id`) |
| `RELEASES` | New evidence releases a blocked condition (requires `target_condition_id`) |
| `REAFFIRMS` | New evidence reaffirms a belief (requires `target_belief_id`) |
| `UNCERTAIN` | New evidence introduces unresolved uncertainty (requires `target_belief_id`) |
| `NO_REVISION` | After inspecting the new evidence, no revision is made (no target ids; still cites the grounding `evidence_id`) |

The **ReTrace-Engine** then resolves the graph and assigns each belief a deterministic **final status**:

```text
A_t(b) = DPA(b, S_t) ∈ {AUTHORIZED, BLOCKED, SUPERSEDED, UNRESOLVED}
precedence:  SUPERSEDES > PREREQUISITE_BLOCK > UNRESOLVED_UNCERTAIN > AUTHORIZED
```

---

## Methods and Baseline Framing

The project evaluates open-weight trainable models against strong baseline alternatives:

1. **Strong LLM DirectJudge**: A baseline that directly predicts the final status from raw dialogues/views, bypassing the ReTrace-Engine. It is used to answer: *Why not directly ask a frontier LLM?*
2. **Prompt-Proposer (Stage A)**: Prompted frontier LLM (e.g., DeepSeek, Gemini) that generates actions over candidate structures. It is used to answer: *Why train a proposer instead of prompting?*
3. **Open-weight ReTrace-Learn model**: Trainable model optimized via SFT (graph extraction & revision proposing), DPA-filtered Rejection Sampling Fine-Tuning (RSFT), and Direct Preference Optimization (DPO).

### Research Map

- **E0: Oracle/Replay Kernel Validation** — hand-authored typed proposals for
  mechanism verification.
- **E1: Fixed-Candidate Revision Evaluation** — evaluates proposer quality over
  identical pre-constructed candidate memory graphs.
- **E2: Stage C Training and Model-Driven Proposal Evaluation** — trains and
  evaluates learning-based proposal policies.
- **E3: Closed-Loop Multi-Agent Workflow** — tests shared-memory effects on
  downstream agent actions and future submissions.
- **E4: STALE/CUPMem External Validation** — validates ReTrace-Learn on external
  stale-memory benchmarks using isolated adapters.

> **Method-paper boundary.** The ReTrace-Learn paper focuses on explicit typed
> memory-revision proposal learning and deterministic DPA verification. Latent
> memory states, delayed-future-utility consolidation, biological memory
> mechanisms, and RL over hidden states are future-scope work.

---

## Repository layout

```text
src/retracemem/                     # the library (importable, paper-facing)
    schemas.py                      # immutable dataclass contracts (nodes/edges)
    tms/ , authorization …          # deterministic core: authorize(), DPA
    multiagent/                     # commit wrappers around authorize()
    proposers/                      # typed_revision_policy (A), replay (C)
    evaluation/multiagent/          # shared evaluation engine:
        config, cases, pipeline,    #   - pipeline = typed proposal → commit → DPA
        directjudge, metrics,       #   - metrics  = pure metric computation
        artifacts, runner, stagec   #   - runner   = Stage A/B; stagec = Stage C
        data/                       #   - dev set builders / dataset export
scripts/                            # thin public python3 entrypoints
configs/                            # reproducible experiment configs
docs/                               # architecture, reproducibility, protocol, positioning
tests/                              # unit / integration / regression tests
experiments/
    multiagent/                     # active workflow helpers + local_training/
    archive/                        # preserved historical / reference code (NOT canonical)
prompts/                            # versioned prompt templates
```

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Live API runs read credentials from a `.env` file in the repo root (e.g.
`SILICONFLOW_API_KEY=...`). Offline modes need no credentials.

---

## Smoke tests & verification

```bash
# Compile check
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests experiments

# Full offline test suite
python3 -m pytest

# End-to-end offline smoke of each method (writes a structured run dir):
python3 scripts/evaluate.py stage-a --mock  --max-cases 2 --output-dir outputs/runs/smoke_a
python3 scripts/evaluate.py stage-b --mock  --max-cases 2 --output-dir outputs/runs/smoke_b
python3 scripts/evaluate.py stage-c --smoke --max-cases 3 --output-dir outputs/runs/smoke_c
```

---

## Running evaluations

Stage A and Stage B are evaluated **jointly** on the same fixed-candidate cases
(fair comparison), so `stage-a` and `stage-b` are aliases into the same A-vs-B
runner, which always reports both.

```bash
# Stage A / Stage B — offline mock
python3 scripts/evaluate.py stage-a --mock

# Stage A / Stage B — live (requires SILICONFLOW_API_KEY in .env)
python3 scripts/evaluate.py stage-a \
    --live --provider siliconflow --model deepseek-ai/DeepSeek-V3 --constrained \
    --output-dir outputs/runs/stageab_dev70

# Stage C — replay decoded adapter/SFT generations through the commit/DPA path
python3 scripts/evaluate.py stage-c \
    --generations-dir path/to/decoded_generations \
    --policy-variant lora_sft --checkpoint-id my_ckpt \
    --output-dir outputs/runs/stagec_dev70

# Stage A on the balanced internal validation set (420 cases)
python3 scripts/evaluate.py stage-a \
    --mock --dataset paper1_balanced --max-cases 10 \
    --constrained --stage-a-variant conflict_aware \
    --output-dir outputs/runs/paper1_balanced_mock10

# Stage C — export the typed-revision SFT dataset (offline)
python3 scripts/export_stagec_data.py
```

Run `python3 scripts/evaluate.py <stage> --help` for the full flag list.

### Provider abstraction (OpenAI-compatible, Anthropic, Ollama)

The evaluation runner is **provider-agnostic**: it never hard-codes one vendor's
request shape. A provider is selected either by registry name (`--provider`) or
by a single-provider config file (`--provider-config`); `--model` stays
authoritative. API keys are resolved from the provider's `api_key_env`
environment variable and are **never** committed.

```bash
# By registry name (backward compatible)
python3 scripts/evaluate.py stage-a --live --provider siliconflow \
    --model deepseek-ai/DeepSeek-V3 --constrained

# By config file (mode/base_url/api_key_env come from the file)
python3 scripts/evaluate.py stage-a --live \
    --provider-config configs/providers/siliconflow.yaml \
    --model deepseek-ai/DeepSeek-V3 --constrained
```

Supported modes: `openai-chat` / `custom-openai-compatible` (SiliconFlow,
DeepSeek, vLLM, SGLang, LM Studio, …), `anthropic-messages` (`/v1/messages`),
and `ollama-chat` (`/api/chat`). See `configs/providers/` for examples and
[`docs/api_providers.md`](docs/api_providers.md) for the full reference.

### Evaluation datasets

`stage-a`/`stage-b` accept `--dataset` (default `dev_expansion`). Both are
internal, deterministically generated sets — neither is an external benchmark:

- **`dev_expansion` (dev70)** — development diagnostic set, 70 cases
  (7 failure types × 2 domains × 5 variants). `--max-cases 400` still loads only 70.
- **`paper1_balanced`** — internal balanced synthetic validation set, 420 cases
  (14 failure types × 2 domains × 15 variants), built programmatically by
  `retracemem.evaluation.multiagent.data.paper1_balanced`. Used for
  ReTrace-Learn internal validation only; not a Stage C training set and not official
  STALE / Memora / CUPMem (those are separate external pathways, not claimed here).

See `docs/experiment_protocol.md` for details.

### Where outputs go

All run artifacts are written under `outputs/` (git-ignored): per-case parsed
records, raw proposer outputs, `dpa_traces.jsonl`, `metrics.json`,
`failure_breakdown.csv`, and a `manifest.json` capturing run provenance.

---

## What is **not** part of ReTrace-Learn

Latent / hidden memory state, learned forgetting, RL consolidation,
delayed-future-utility learning, and biological-memory mechanisms belong to
future-scope work and must not appear in this codebase. STALE/CUPMem is an
external validation/baseline pathway, not the definition of the method.

## Legacy / archived paths

Legacy and archived research code (historical action ablations, composition studies, E4 STALE/CUPMem external validation, and older comparison/diagnostic runners) has been removed from the active tree to maintain a clean research artifact. These files remain fully recoverable from Git history.

---

## Governance

Generated summaries and run logs live under `outputs/` and are excluded from git.
Checkpoints, adapter weights, API caches, benchmark downloads, and API keys are
never committed. Core DPA logic stays API-free and deterministic; benchmark- and
provider-specific logic lives in proposers / runners / adapters. See `AGENTS.md`
for the full contributor contract.

# MemPatch Revision Module

**Paper:** MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents

## 1. Problem: Rapid Memory Integration (RMI)

LLM agents operating with long-term memory receive new evidence over time. RMI requires the agent to determine, for each visible memory, whether it remains `current`, is `outdated`, `blocked`, `unresolved`, `out_of_scope`, or `restored` — rather than blindly appending new information.

MemPatch turns memory revision into a **constrained benchmark-compatible state-transition problem** rather than free-form answer generation.

## 2. Module: MemPatch Revision Module

The **MemPatch Revision Module** is not a Transformer block or LoRA adapter. It is an algorithm module on the agent memory state-transition layer.

**Core idea:** Given a `scenario` / `event_trace`, the model does not answer freely. It first produces a benchmark-compatible revision response; DPA performs consistency authorization and projection; the final output is scored by MemPatch-Bench on `decision`, `memory_state`, `evidence_event_ids`, `failure_diagnosis`, and `answer`.

**One sentence:** MemPatch turns memory revision into a constrained benchmark-compatible state-transition problem rather than free-form answer generation.

**DPA role:** The model proposes; DPA authorizes; the benchmark evaluates the resulting memory state. DPA is a deterministic verifier inside the module, not a separate paper or framework.

## 3. Inputs

| Symbol | Description |
|--------|-------------|
| `S` | Scenario: `public_input`, `event_trace`, query / task prompt |
| `M` | Visible memory candidates (`initial_memory`, related beliefs) |
| `πθ` | Revision Response Policy (learned or API) |
| `A` | Deterministic DPA verifier (`authorize(...)`) |

## 4. Outputs

Benchmark-compatible `response` / `r_final`:

```json
{
  "decision": "use_current_memory",
  "memory_state": {"m1": "current", "m2": "outdated"},
  "evidence_event_ids": ["e2", "e5"],
  "failure_diagnosis": "stale_memory_reuse",
  "answer": "..."
}
```

## 5. Internal roles

These are **implementation roles inside one module**, not separate paper contributions:

| Role | Maps to | Code |
|------|---------|------|
| **Scenario View Builder** | `S, M` → revision view `V` | `src/retrace_learn/runtime/graph_extractor.py` |
| **Revision Response Policy** | `V` → raw response `r_raw` | `src/retrace_learn/runtime/learned_proposer.py` |
| **DPA-Consistent Projection** | parse + authorize + project to legal transitions | `dpa_runtime.py`, `authorization.py`, `benchmark_projection.py` |
| **Benchmark-grounded Feedback** | metrics → training signal | `src/retrace_learn/runtime/reward.py` |

Implementation entrypoints:

- Full method path: `scripts/run_mempatch_revision_module.py`
- Direct Response baseline: `scripts/run_retrace_bench_model.py` (alias: `run_mempatch_model.py`)
- Evaluator: `scripts/evaluate_retrace_bench_predictions.py` (alias: `evaluate_mempatch_predictions.py`)

## 6. Algorithm 1 — MemPatch Revision Module

```text
Input:
  scenario S = (public_input, event_trace, query)
  visible memory candidates M
  revision response policy πθ
  deterministic DPA verifier A

Step 1 — Scenario View Builder:
  V ← BuildScenarioRevisionView(S, M)

Step 2 — Revision Response Policy:
  r_raw ← πθ(V)

Step 3 — Parse:
  a ← ParseRevisionResponse(r_raw)

Step 4 — DPA-Consistent Projection:
  T ← DPAConsistentProjection(A, a, V)
      // RevisionGate + authorize; enforce evidence, scope,
      // temporal validity, typed transition constraints

Step 5 — Benchmark response projection:
  r_final ← ProjectToBenchmarkResponse(T, r_raw)

Output:
  r_final = {
    decision,
    memory_state,
    evidence_event_ids,
    failure_diagnosis,
    answer
  }
```

**Evaluation:** MemPatch-Bench scores `r_final` against `hidden_gold` (never shown to the policy at inference).

## 7. Training objective

Supervised decomposition (when gold labels are available):

```text
L = L_state + L_evidence + L_decision + L_diagnosis
```

| Term | Target |
|------|--------|
| `L_state` | `memory_state` vs `hidden_gold.expected_memory_state` |
| `L_evidence` | `evidence_event_ids` vs `hidden_gold.expected_evidence_event_ids` |
| `L_decision` | `decision` vs `hidden_gold.expected_decision` |
| `L_diagnosis` | `failure_diagnosis` vs `hidden_gold.expected_failure_diagnosis` |

**Preference / DPO-style pairs** (when configured): construct positives and negatives from benchmark response quality.

Positive response:

- correct `memory_state`, `evidence_event_ids`, `decision`
- no stale reuse
- correct or acceptable `failure_diagnosis`

Negative response (benchmark failure modes):

- `stale_memory_reuse`, `under_update`, `over_update`
- `scope_leakage`, `wrong_source_attribution`, `memory_hallucination`

**Benchmark-grounded reward** (implemented in `reward.py`; weights tunable):

```text
R = memory_state_accuracy
  + evidence_f1
  + joint_revision_success
  - stale_reuse_rate
  - over_update_penalty
  - under_update_penalty
  - scope_leakage_penalty
```

Benchmark-grounded feedback can support SFT, RSFT, or DPO-style policy improvement when training scripts are configured. Full DPO pipelines are not claimed unless corresponding scripts and results exist.

## 8. Ablation plan

| Variant | What is removed / changed |
|---------|---------------------------|
| **Full MemPatch Revision Module** | All four roles |
| **w/o DPA-Consistent Projection** | Skip Step 4; use raw policy output (`bypass_gate` / no `authorize`) |
| **w/o explicit memory_state prediction** | Direct answer only; no structured `memory_state` field |
| **w/o evidence grounding** | No `evidence_event_ids` constraint or reward term |
| **Direct answer only** | Baseline: free-form answer, no revision module |
| **Response policy w/o benchmark-grounded feedback** | Fixed or prompted policy; no `reward.py` training signal |

Baselines in repo configs: `ReTrace-Prompt`, `DirectJudge-API`, MemPatch-Bench model runner (end-task `response` without full module).

## 10. Cost-aware experiment plan (P2)

Do not start with full 3500 rows or closed-source flagship models.

| Stage | Scope | Goal |
|-------|-------|------|
| **0** | `compileall` + evaluator strict-mode tests | No model spend; schema / projection sanity |
| **1** | `main20` + `hard20`, one cheap open-weight model | Smoke Direct Response vs Revision Module runner |
| **2** | `main80` + `hard20`, up to three open models | Small comparison only |
| **3** | `main200` + `hard100` | Direct baseline vs full module vs w/o DPA projection |
| **4** | Full `main3000` + `hard500` | Only after Stage 3 shows signal |

Example Revision Module smoke:

```bash
python scripts/run_mempatch_revision_module.py \
  --data local/MemPatch/main/scenarios.jsonl \
  --out-predictions local/predictions/mempatch_main20.jsonl \
  --max-cases 20 \
  --resume
```

Public HF release metadata (`hf_release/mempatch_v1_1/manifest.json`) documents **main=3000, hard=500, total=3500**. Scenario JSONL files are not vendored in git; download from Hugging Face into `local/`.

## 9. Paper-facing summary

MemPatch-Bench defines the benchmark-compatible response interface for RMI. The MemPatch Revision Module learns to produce better revision responses. DPA projects proposed revisions into legal memory-state transitions. Benchmark-grounded feedback improves the response policy.

# MemPatch Agent Instructions

**Read order:** `AGENTS.md` → `README.md` → `docs/mempatch_revision_module.md`

## Unified paper

**MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents**

RMI: integrate new evidence into correct `memory_state` labels — not blind append.

One story:

- **MemPatch-Bench** — paper-facing `response` interface and `hidden_gold` scoring
- **MemPatch Revision Module** — algorithm module that produces benchmark-compatible revision responses
- **DPA** — deterministic verifier inside the module (`authorize`); not a separate framework
- **Benchmark-grounded feedback** — training signal from benchmark metrics

Do not present MemPatch-Bench and the Revision Module as two papers or tracks.

## MemPatch Revision Module

State-transition layer algorithm (not a Transformer block). See `docs/mempatch_revision_module.md` for Algorithm 1.

```text
V ← BuildScenarioRevisionView(S, M)
r_raw ← πθ(V)
a ← ParseRevisionResponse(r_raw)
T ← DPAConsistentProjection(A, a, V)
r_final ← ProjectToBenchmarkResponse(T, r_raw)
```

**Internal roles** (not paper contributions):

1. Scenario View Builder — `graph_extractor.py`
2. Revision Response Policy — `learned_proposer.py`
3. DPA-Consistent Projection — `dpa_runtime.py`, `retracemem/authorization.py`
4. Benchmark-grounded Feedback — `reward.py`

## Benchmark response interface

- `response.decision`, `response.memory_state`, `response.evidence_event_ids`, `response.failure_diagnosis`, `response.answer`
- Gold: `hidden_gold.expected_*` (canonical v1.1 only; no legacy field fallbacks)

## DPA

The model proposes; DPA authorizes; the benchmark evaluates `memory_state`. Call only `authorize(...)` — not DPA or RevisionGate directly.

## Baselines

- `ReTrace-Prompt` — typed actions over fixed revision view → DPA projection
- `DirectJudge-API` — direct status prediction, bypasses projection
- `retrace_learn` config — full MemPatch Revision Module

## Training

Policy sees method-visible inputs only (revision view, evidence, candidates). Never `hidden_gold` at inference.

```text
L = L_state + L_evidence + L_decision + L_diagnosis
```

Benchmark-grounded feedback can support SFT / RSFT / DPO-style improvement when scripts exist. Do not overclaim full DPO without training scripts and results.

## Experiment tiers

E0 oracle kernel · E1 fixed-candidate eval · E2 Revision Module training · E3 multi-agent loop · E4 STALE external validation

## Do not drift

No duplicate `authorize`, no STALE gold in method inputs, no generic RAG clone, no DPA semantic changes without demonstrated bugs.

## Clean worktree

Do not commit `local/`, `artifacts/`, caches, checkpoints, predictions, API keys.

```bash
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf .pytest_cache .pycache_compile
```

## Verification

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q benchmark scripts src
```

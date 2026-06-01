# ReTrace-Bench

**ReTrace-Bench** is an evaluation benchmark package designed specifically to evaluate room-for-revision, conflict resolution, and defeat-path authorization (DPA) mechanisms in shared-memory LLM agents.

## Methodology & Conceptual Boundary

ReTrace-Bench is strictly **evaluation-only** and independent of the method paper framework (ReTrace-Learn).
- **ReTrace-Learn**: The trainable method (SFT/RL/DPO) that trains Graph Extractors and Typed Revision Proposers. Uses internal data under `data/retrace_learn/`.
- **ReTrace-Bench**: The independent benchmark paper evaluating different agent memory updating strategies. The evaluation data under `data/retrace_bench/` must **never** be used for training or validation tuning of ReTrace-Learn.

---

## Benchmark Design Target

The full version target is:
- **2,500** scenarios
- **10,000** probe queries (exactly 4 per scenario)
- **6** domains:
  - `coding_agent_debugging`
  - `research_agent_memory`
  - `personal_preference_memory`
  - `calendar_workflow`
  - `tool_use_assistant`
  - `multi_agent_knowledge_base`
- **4** probe types:
  - `state_resolution`
  - `premise_resistance`
  - `policy_adaptation`
  - `audit_localization`
- **7** revision families:
  - `supersedes`, `blocks`, `releases`, `uncertain`, `reaffirms`, `no_revision`, `mixed_multi_action`

Currently, a **100-scenario smoke version** is supported.

---

## Protocols

1. **raw-only**: dialogue history + memory snapshot + query -> answer. The main public protocol.
2. **structured revision**: dialogue + memory snapshot + candidate graph -> proposed actions.
3. **oracle diagnostic**: evaluates using gold revision links. Used for error attribution.

---

## CLI Usage

### 1. Build Smoke Data
```bash
python scripts/build_retrace_bench_v1.py \
  --out data/retrace_bench/v1_smoke \
  --num-scenarios 100 \
  --queries-per-scenario 4 \
  --seed 7
```

### 2. Validate Data
```bash
python scripts/validate_retrace_bench.py --data data/retrace_bench/v1_smoke
```

### 3. Run Baselines
```bash
python scripts/run_retrace_bench_baseline.py \
  --data data/retrace_bench/v1_smoke \
  --baseline latest_only \
  --out outputs/retrace_bench/latest_only_smoke.jsonl
```

### 4. Aggregate Scores
```bash
python scripts/aggregate_retrace_bench_results.py \
  --predictions outputs/retrace_bench/latest_only_smoke.jsonl \
  --out outputs/retrace_bench/latest_only_smoke_report.json
```

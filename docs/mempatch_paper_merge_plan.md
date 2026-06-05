# MemPatch paper outline

**Title:** MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents

**Authority:** `docs/mempatch_revision_module.md` defines the algorithm module.

## Narrative

MemPatch-Bench defines the benchmark-compatible `response` interface. The MemPatch Revision Module learns better revision responses. DPA projects proposals into legal memory-state transitions. Benchmark-grounded feedback improves the policy.

## Section map

| § | Content |
|---|---------|
| 1 Intro | RMI, Revision Module, contributions (bench + module + learning) |
| 2 Related | Agent memory, knowledge update, revision benchmarks |
| 3 Problem | RMI formulation, Algorithm 1, typed edges |
| 4 MemPatch-Bench | Scenarios, splits, `hidden_gold`, metrics |
| 5 MemPatch Revision Module | Four internal roles, DPA projection, training objective |
| 6 Experiments | Baselines, ablations (see revision module §8) |
| 7 Results | Headline metrics from `scorers_general.py` |
| 8 Analysis | Per-`failure_mode` breakdown, limitations |

## Terminology

| Paper term | Implementation |
|------------|----------------|
| MemPatch-Bench | `benchmark/retrace_bench/` |
| MemPatch Revision Module | `src/retrace_learn/` + `src/retracemem/` commit path |
| Scenario View Builder | `graph_extractor.py` |
| Revision Response Policy | `learned_proposer.py` |
| DPA-Consistent Projection | `dpa_runtime.py`, `authorize` |
| Benchmark-grounded Feedback | `reward.py` |

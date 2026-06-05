# MemPatch paper outline

**Title:** MemPatch: Benchmarking and Improving Rapid Memory Integration in LLM Agents

One unified paper. MemPatch-Bench defines the problem and evaluation interface; the MemPatch scaffold learns benchmark-compatible responses; DPA authorizes internally; benchmark-grounded feedback trains the policy.

## Core narrative

**RMI:** Given `scenario` + `event_trace` + prior memories, produce the correct `memory_state`, grounded `evidence_event_ids`, `decision`, and `failure_diagnosis`.

**Paper-facing interface** — benchmark `response`:

```json
{
  "decision": "use_current_memory",
  "memory_state": {"m1": "current", "m2": "outdated"},
  "evidence_event_ids": ["e2"],
  "failure_diagnosis": "stale_memory_reuse",
  "answer": "..."
}
```

**Scaffold pipeline** (implementation roles, not separate contributions):

| Role | Code |
|------|------|
| Scenario View Builder | `src/retrace_learn/runtime/graph_extractor.py` |
| Revision Response Policy | `src/retrace_learn/runtime/learned_proposer.py` |
| Benchmark-grounded feedback | `src/retrace_learn/runtime/reward.py` |
| DPA / commit | `src/retracemem/authorization.py`, `tms/*` |

## Section map

| § | Content | Sources |
|---|---------|---------|
| 1 Intro | RMI motivation, contributions (bench + scaffold + learning) | `README.md`, HF README |
| 2 Related | Agent memory, editing, conflict benchmarks | taxonomy |
| 3 Problem | Typed edges, DPA precedence, `authorize` flow | `AGENTS.md`, `schemas.py` |
| 4 MemPatch-Bench | Scenarios, splits, `hidden_gold`, metrics | `hf_release/`, `scorers_general.py` |
| 5 Scaffold | View Builder → Response Policy → DPA commit | `retrace_learn/runtime/*` |
| 6 Setup | Baselines, splits, metrics table | `scripts/`, `api.py` |
| 7 Results | Headline metrics, ablations | `local/results/` |
| 8 Analysis | Failure modes, limitations | taxonomy |

## Terminology (canonical)

| Concept | Term |
|---------|------|
| Evaluation layer | MemPatch-Bench |
| Method implementation | MemPatch scaffold |
| Event input | `scenario`, `event_trace`, `public_input` |
| Gold labels | `hidden_gold` |
| Model output | `response` |
| Internal graph | revision view |
| Training signal | benchmark-grounded feedback |

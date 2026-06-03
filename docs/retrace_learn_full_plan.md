# ReTrace-Learn: Raw Dialogue Revision Authorization Protocol

This document details the system design, interfaces, and data contracts for **ReTrace-Learn**. It bridges raw dialogue inputs and the deterministic **ReTrace-Engine** revision authorization kernel.

## 1. Protocol Architecture

ReTrace-Learn upgrades the shared-memory revision authorization process by removing the requirement for pre-constructed candidate views. It replaces it with a learned extraction layer:

```
+--------------------------------------------------------------------------+
|                       Raw Dialogue Protocol (ReTrace-Learn)              |
+--------------------------------------------------------------------------+
| 1. Raw Dialogues / Subagent Submissions (Text)                           |
|    |                                                                     |
|    v                                                                     |
| 2. Graph Builder (Stage 1, learned: SFT)                                 |
|    |                                                                     |
|    +--> Extracted Candidate Memory Graph (Dict)                          |
|         - evidence_nodes, belief_nodes, condition_nodes,                 |
|           candidate_replacement_beliefs, dependency_edges                |
|    |                                                                     |
|    v                                                                     |
|    Bridge Wrapper (implementation detail: runtime/views.py)              |
|    |                                                                     |
|    +--> SharedCandidateView (Dataclass)                                  |
|                                                                          |
| 3. Proposal Policy (Stage 2, learned: SFT / DPA-guided RSFT/DPO)         |
|    |                                                                     |
|    +--> Proposed Revision Actions (SUPERSEDES, BLOCKS, etc.)             |
|    |                                                                     |
|    v                                                                     |
|    ReTrace-Engine commit path (deterministic; impl detail of stages 2-3)  |
|    - Parser                                                              |
|    - RevisionGate (Structural Admission)                                 |
|    - Defeat-Path Authorization (DPA)                                     |
|    |                                                                     |
|    +--> Final Auditable Belief Statuses + Defeat Paths                   |
+--------------------------------------------------------------------------+
```

---

## 2. Fixed-Candidate Protocol vs. Raw-Dialogue Protocol

### Fixed-Candidate Protocol
- **Input**: A pre-constructed candidate view specifying beliefs, condition anchors, and dependency edges.
- **Task**: The proposer only decides the *revision actions* (e.g., `SUPERSEDES(b1, b2)`). It does not need to parse raw text to construct the belief nodes or conditions.
- **Purpose**: Serves as a controlled baseline to isolate and compare revision decision-making capabilities.

### Raw-Dialogue Protocol
- **Input**: A chronological sequence of raw dialogues, agent utterances, or subagent tool submissions.
- **Task**: 
  1. Build the memory graph elements (beliefs, conditions, dependency anchors, evidence text) from raw dialogue.
  2. Assemble a temporal candidate view.
  3. Propose typed revision actions.
  4. Pass everything to the deterministic ReTrace-Engine.
- **Purpose**: Represents the full end-to-end multi-agent shared-memory maintenance pipeline.

---

## 3. Strict No-Gold-Leakage Boundary

To guarantee evaluation validity and prevent data leakage:
1. **Method-Visible Constraint**: During SFT, RSFT, DPO, and inference, the learned policy sees *only* method-visible inputs (such as the output of the Graph Builder and the new evidence). It is completely blind to gold actions or DPA final statuses.
2. **Evaluator Isolation**: Gold final statuses reside strictly within the evaluation boundary and are used solely to compute rewards or score outputs.
3. **No-DirectJudge Fallback**: The learned proposer MUST ONLY emit typed revision actions from the canonical vocabulary. It is strictly prohibited from predicting final DPA statuses (`AUTHORIZED`, `SUPERSEDED`, `BLOCKED`, `UNRESOLVED`) directly, preserving the deterministic nature of the `ReTrace-Engine`.

---

## 4. Graph Builder Target Schema

The target output of the learned **Graph Builder** must serialize to a strict JSON structure matching the following key mapping:

```json
{
  "evidence_nodes": [
    {
      "evidence_id": "ev_1",
      "session_id": "sess_1",
      "timestamp": "2026-06-01T00:00:00Z",
      "text": "Subagent A submitted a report showing the user relocated to Seattle.",
      "source_dataset": "subagent_dialogue",
      "source_pointer": "utterance_14"
    }
  ],
  "belief_nodes": [
    {
      "belief_id": "b_1",
      "proposition": "The user is located in Seattle.",
      "source_evidence_ids": ["ev_1"]
    }
  ],
  "condition_nodes": [
    {
      "condition_id": "c_1",
      "scope_id": "Seattle",
      "text": "Seattle office lease remains active."
    }
  ],
  "candidate_replacement_beliefs": [
    {
      "belief_id": "b_2",
      "proposition": "The user is located in Tacoma.",
      "source_evidence_ids": ["ev_2"]
    }
  ],
  "dependency_edges": [
    {
      "edge_id": "dep_1",
      "belief_id": "b_1",
      "condition_id": "c_1",
      "inducer": "location_service",
      "edge_type": "REQUIRES"
    }
  ]
}
```

This schema is strictly validated by `retrace_learn.schemas.validate_memory_graph`.

---

## 5. First Full Experiment Loop

We have implemented the first complete, runnable evaluation pipeline. It provides deterministic components to execute the full evaluation loops of Protocol A (Fixed-Candidate) and Protocol B (Raw-Dialogue) on a synthetic dataset.

### Reproduction Commands
To run the full pipeline and output metric JSON summaries:

python scripts/build_raw_dialogue_synth.py --out outputs/smoke/raw_dialogue_synth.jsonl --n 50 --seed 7

python scripts/run_fixed_candidate_matrix.py --input outputs/smoke/raw_dialogue_synth.jsonl --out outputs/smoke/fixed_candidate_metrics.json

python scripts/run_raw_dialogue_matrix.py --input outputs/smoke/raw_dialogue_synth.jsonl --out outputs/smoke/raw_dialogue_metrics.json

python -m pytest -q

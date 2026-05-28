# Stage C Go/No-Go Report: Deferred Learned Local Verifier

**Date**: 2026-05-28  
**Phase**: V1-4 Final Integration  
**Recommendation**: **NO-GO** (Defer Stage C Development)

---

## 1. Decision Summary

In alignment with the repository execution contract and the ICLR 2027 Paper 1 blueprint, we recommend a strict **NO-GO** on starting Stage C (*ReTrace-Local*: a deferred learned local typed-edge verifier) at this juncture. 

Stage C development should be deferred until the prerequisites defined in Section 3 are fully satisfied. The current v1 phase has successfully validated the deterministic DPA routing and established clean-room adapters and evaluators for the official benchmarks (STALE and Memora). Initiating Stage C now would be premature and counter-productive to scientific rigor.

---

## 2. Rationale for Deferral

1. **Lack of Verified Ground-Truth Evidence Graphs**: 
   Stage C requires training or fine-tuning a local semantic model (or small language model) to predict local typed evidence/dependency edges directly. Currently, we lack a large-scale, high-quality, human-audited corpus of ground-truth DPA edge graphs. Training on unverified outputs of Stage A verifiers would propagate semantic errors and compromise DPA's safety guarantees.
   
2. **Infrastructure Verification First**: 
   We have just completed the official evaluation adapters and run pathways (STALE & Memora). We must first run extensive evaluations on the Stage A / Stage B baseline to map the exact performance envelope of LLM-based verification vs. direct LLM adjudication.
   
3. **No Performance Baseline for Local Verifiers**:
   Without complete empirical data from the frozen v1 baseline (Stage A vs. Stage B) on full benchmark tracks, we cannot formulate a concrete target accuracy or compute latency budgets for a local verifier (Stage C).

---

## 3. Prerequisites for Stage C Commencement

Before transitioning to Stage C, the following conditions must be met:

1. **Complete Frozen Evaluation Results**:
   Successfully execute the frozen evaluation pathways (via `run_stale_official_eval.py` and `run_memora_official_eval.py`) in `--live` mode across all evaluation personas/periods to produce a definitive comparison between Stage A (`ReTrace-LLM`) and Stage B (`DirectJudge-LLM`).
   
2. **Gold Standard Dataset Extraction**:
   Extract at least 1,000 high-confidence, audited trace logs of `SharedCandidateView` along with their corresponding admitted `EvidenceEdge` and `DependencyEdge` structures from the Stage A live runs.
   
3. **Distillation/Fine-tuning Pipeline**:
   Establish a reproducible training pipeline (e.g., via LoRA or full parameter fine-tuning on LLama-3-8B-Instruct or smaller models) and define clear evaluation metrics for edge-prediction classification accuracy.
   
4. **Safety Boundary Definition**:
   Define strict fallback rules when the local verifier emits an `UNCERTAIN` edge type, ensuring it defaults gracefully to a stronger remote LLM verifier (hybrid Stage A/C execution).

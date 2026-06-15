# Audit: Figure Consistency and Ontology Alignment

This report checks the consistency between the figures planned/present in the paper and the actual repository implementation, proposing concrete fixes.

## Figure 1: Motivation & Contrast
*   **Current Issue:** Uses un-aligned terminology and obsolete states (such as `Superseded` or `Safe`).
*   **Ontology Alignment:**
    *   Align with the official DPA statuses in the code: `AUTHORIZED`, `BLOCKED`, `SUPERSEDED`, `UNRESOLVED`.
    *   Replace `Safe` with `Authorized Commit`.
    *   Remove references to DPA from the primary motivation to focus on the immediate-answer/persistent-state contrast.

## Figure 2: Benchmark Pipeline & Evaluation
*   **Current Issue:** Depicts the `Overall Joint` evaluator as the headline metric.
*   **Alignment & Re-design:**
    *   Replace `Overall Joint` with the **Reference Transition Semantics** that yield a **Unique Gold Normal Form**.
    *   Depict the pipeline: Public View $\rightarrow$ System $\rightarrow$ Error Signature / Failure Fingerprint.
    *   Incorporate the Controlled Corruptions $\rightarrow$ Sensitivity Matrix $\rightarrow$ Identifiability Audit.
    *   Include a schematic sensitivity matrix heatmap directly in Figure 2.

## Figure 3: MemPatch Runtime
*   **Current Issue:** Depicts consensus mode as the mandatory path.
*   **Runtime Realignment:**
    *   Redraw to show **Compatibility Mode** (a single frozen/adapted predictor feeding into the compiler) as the primary execution path.
    *   Represent **Consensus Mode** (multiple predictors with deterministic reconciliation) as an optional branch.
    *   Show the runtime flow: Predictor $\rightarrow$ Compiler $\rightarrow$ Guard $\rightarrow$ Canonical Commit $\rightarrow$ Audit Certificate.

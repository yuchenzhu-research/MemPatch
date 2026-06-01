# Memory Reliability Taxonomy

This document outlines the standard memory reliability taxonomy in ReTrace-Bench v2. The taxonomy is divided into generic memory lifecycle operations and optional structured-revision diagnostics.

## Generic Memory Lifecycle Operations

These operations describe the fundamental lifecycle transitions of a memory entry in an agent's memory system. They are model-agnostic and apply to any agent memory architecture:

| Operation | Description | Example Failure Mode |
| :--- | :--- | :--- |
| **acquire** | Write a new memory entry from an observation or interaction. | Memory hallucination, over-update. |
| **update** | Modify an existing memory entry to reflect new information. | Under-update, stale memory reuse. |
| **invalidate**| Flag a memory entry as no longer valid or applicable. | Stale memory reuse, conflict collapse. |
| **retain** | Explicitly keep a memory entry unchanged despite new decoy observations. | Spurious over-update. |
| **ignore** | Deliberately skip recording irrelevant, noisy, or redundant information. | Unnecessary memory write. |
| **forget** | Safely delete or prune memories to comply with privacy or space constraints. | Failure to forget, policy violation. |
| **merge** | Synthesize conflicting or duplicate observations into a single coherent state. | Conflict collapse, state merge error. |
| **restrict** | Limit memory visibility or access based on role, scope, or context. | Scope leakage, policy violation. |
| **restore** | Re-activate a previously invalidated or blocked memory entry. | Failure to release or restore. |

## Diagnostic Structured-Revision Vocabulary (Optional)

For systems that implement structured revision (like ReTrace-Learn), the following typed revision vocabulary is supported as an optional diagnostic track. **Participants are not forced to output these typed actions for the main track.**

- **SUPERSEDES**: Newly observed evidence directly replaces or updates a prior memory entry. Requires a grounded `replacement_belief_id`.
- **BLOCKS**: Newly observed evidence invalidates an existing memory entry because its prerequisite conditions are no longer met.
- **RELEASES**: Newly observed evidence restores a previously blocked memory entry by satisfying its blocked conditions.
- **UNCERTAIN**: Newly observed evidence makes an existing memory entry doubtful, lowering its status to unresolved.
- **REAFFIRMS**: Newly observed evidence confirms and validates an existing memory entry against a potential conflict.
- **NO_REVISION**: No changes are required to the existing memory structure.

Cite v1 design pillars and note that v2 extends them:
- **Topology Awareness**: Tracking dependency structures between beliefs and conditions.
- **Path Precedence**: Resolving conflicts chronologically or logically using typed defeat-paths.
- **Auditability**: Localizing decisions to explicit groundable source event traces.
- *v2 Extension*: Beyond DPA-centric graphs, v2 focuses on black-box behavioral correctness, multi-agent synchrony, scope limits, and forgetting compliance.

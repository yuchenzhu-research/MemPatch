# ReTrace-Learn — Method Paper Workspace

This folder is the working index for the ReTrace-Learn method paper. It points
to the canonical method docs and runtime surfaces without duplicating code,
data, or experiment outputs.

ReTrace-Learn studies trainable typed memory-revision proposal for
multi-agent/subagent shared memory. Learned modules extract candidate graph
state and propose typed actions; the deterministic Authorization Court
(ReTrace-Engine, `authorize(...)`) performs admission and final status
computation.

## What is in this folder

- `paper_skeleton.md` — section-by-section skeleton for the method paper.
- `section_bank.md` — compact prose and source pointers for each section.
- `figure_plan.md` — planned figures and their canonical inputs.

## Canonical sources

- Architecture: `docs/architecture.md`
- Pipeline specification: `docs/retrace_learn_pipeline.md`
- Experiment protocol: `docs/experiment_protocol.md`
- Method positioning: `docs/research_positioning.md`,
  `docs/retrace_learn_positioning.md`
- Learned modules: `src/retrace_learn/`
- Authorization Court / ReTrace-Engine: `src/retracemem/`
- Public evaluation entrypoint: `scripts/evaluate.py`

# ReTrace-Learn

ReTrace-Learn is the method-paper track for trainable shared-memory revision
authorization. It learns to extract structured graph state and propose typed
revision actions, while the deterministic Authorization Court
(`authorize(...)`, implemented by ReTrace-Engine) performs all admission and
final status computation.

## Canonical Locations

- Learned method modules: `src/retrace_learn/`
- Deterministic Authorization Court: `src/retracemem/`
- Method docs: `docs/architecture.md`, `docs/retrace_learn_pipeline.md`,
  `docs/experiment_protocol.md`
- Method paper workspace: `papers/retrace_learn/`
- Training/evaluation scripts: `scripts/evaluate.py`,
  `scripts/export_stagec_data.py`

## Quick Commands

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q src tests experiments

python3 scripts/evaluate.py stage-a --mock --max-cases 2 \
  --output-dir outputs/runs/smoke_a

python3 scripts/evaluate.py stage-c --smoke --max-cases 3 \
  --output-dir outputs/runs/smoke_c
```

For the method boundary and experiment hierarchy, start with
[`docs/experiment_protocol.md`](docs/experiment_protocol.md).

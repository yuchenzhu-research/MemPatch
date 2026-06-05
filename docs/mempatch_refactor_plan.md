# MemPatch refactor plan (archived)

Planning notes only. The active narrative is unified in `README.md` and `AGENTS.md`.

**Current state:** One MemPatch paper. Package paths remain `benchmark/retrace_bench/`, `src/retrace_learn/`, `src/retracemem/` as implementation locations — not paper concepts.

**Do not execute** large physical renames without an explicit follow-up task. Future optional moves (`mempatch/` facade) are deferred.

**Verification:**

```bash
env PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m compileall -q benchmark scripts src
```

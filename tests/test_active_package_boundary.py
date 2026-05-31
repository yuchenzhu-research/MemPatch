"""Regression guard: the active package must not depend on runner scripts or
archived experiment code.

Paper 1 cleanup invariant: shared logic lives in ``src/retracemem/`` and Stage
A/B/C plug into it. The active package must never import:

- any ``experiments.*`` module (runner scripts / archived code), or
- a sibling ``run_*`` evaluation script.

If this test fails, move the shared logic into ``retracemem`` instead of
importing it from a script or from ``experiments/``.
"""
from __future__ import annotations

import ast
import pathlib

SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "retracemem"


def _iter_imported_modules(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                yield node.module


def test_active_package_does_not_import_experiments_or_runner_scripts():
    offenders: list[str] = []
    for py in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        rel = py.relative_to(SRC_ROOT.parents[1])
        for mod in _iter_imported_modules(tree):
            top = mod.split(".")[0]
            leaf = mod.split(".")[-1]
            if top == "experiments":
                offenders.append(f"{rel}: imports '{mod}' (experiments.*)")
            elif leaf.startswith("run_") or "run_stageab" in mod or "run_stagec" in mod:
                offenders.append(f"{rel}: imports runner script '{mod}'")
    assert not offenders, "Active package must not import experiments/runner scripts:\n" + "\n".join(offenders)

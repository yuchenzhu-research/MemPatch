"""Shared path helpers for scripts/ CLIs."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
MEMPATCH_DIR = REPO_ROOT / "mempatch"
# Back-compat symlink for local-only CUDA scripts: src -> mempatch
SRC_DIR = REPO_ROOT / "src"


def repo_root_for(caller: str | Path) -> Path:
    """Return the repository root for a script living under scripts/."""
    path = Path(caller).resolve()
    scripts = path.parent
    while scripts.name != "scripts":
        parent = scripts.parent
        if parent == scripts:
            raise RuntimeError(f"expected script under scripts/: {path}")
        scripts = parent
    return scripts.parent


def bootstrap_from(caller: str | Path, *, src: bool = False) -> tuple[Path, Path]:
    """Insert repo root, scripts/, and optionally mempatch/ on sys.path."""
    repo = repo_root_for(caller)
    scripts = repo / "scripts"
    for entry in (repo, scripts):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)
    if src:
        for subdir in (MEMPATCH_DIR, SRC_DIR):
            text = str(subdir)
            if subdir.exists() and text not in sys.path:
                sys.path.insert(0, text)
    return repo, scripts

"""Shared path helpers for tools/ CLIs."""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
MEMPATCH_DIR = REPO_ROOT / "mempatch"


def repo_root_for(caller: str | Path) -> Path:
    """Return the repository root for a CLI living under tools/."""
    path = Path(caller).resolve()
    tools = path.parent
    while tools.name != "tools":
        parent = tools.parent
        if parent == tools:
            raise RuntimeError(f"expected CLI under tools/: {path}")
        tools = parent
    return tools.parent


def bootstrap_from(caller: str | Path, *, src: bool = False) -> tuple[Path, Path]:
    """Insert the repository paths needed by a tool CLI.

    ``src`` is retained for compatibility with older callers; the project now
    has one package root, so it only enables the local ``mempatch`` directory.
    """
    repo = repo_root_for(caller)
    tools = repo / "tools"
    for entry in (repo, tools):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)
    if src:
        for subdir in (MEMPATCH_DIR,):
            text = str(subdir)
            if subdir.exists() and text not in sys.path:
                sys.path.insert(0, text)
    return repo, tools

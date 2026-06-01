"""Training entrypoints for ReTrace-Learn (LoRA SFT, DPO, GRPO).

Heavy ML dependencies (torch / transformers / peft / trl) are imported *lazily*
inside the ``train_*`` functions so this package imports cleanly in a CPU-only,
dependency-light environment. The dataset-building halves are pure-stdlib and
fully runnable/testable via ``--dry-run``.
"""
from pathlib import Path
from typing import Any


def check_contamination(obj: Any) -> None:
    """Recursively reject any path/string pointing to data/retrace_bench."""
    if isinstance(obj, (str, Path)):
        s = str(obj).replace("\\", "/")
        if "data/retrace_bench" in s:
            raise RuntimeError(
                "ReTrace-Bench is evaluation-only and must not be used for ReTrace-Learn training."
            )
    elif isinstance(obj, dict):
        for k, v in obj.items():
            check_contamination(k)
            check_contamination(v)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            check_contamination(item)


"""Training entrypoints for ReTrace-Learn (LoRA SFT, DPO, GRPO).

Heavy ML dependencies (torch / transformers / peft / trl) are imported *lazily*
inside the ``train_*`` functions so this package imports cleanly in a CPU-only,
dependency-light environment. The dataset-building halves are pure-stdlib and
fully runnable/testable via ``--dry-run``.
"""
from pathlib import Path


def check_contamination(path: str | Path) -> None:
    path_str = str(path)
    if "data/retrace_bench" in path_str or "data/retrace_bench" in path_str.replace("\\", "/"):
        raise RuntimeError(
            "ReTrace-Bench is evaluation-only and must not be used for ReTrace-Learn training."
        )

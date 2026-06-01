from pathlib import Path


def check_contamination(path: str | Path) -> None:
    """Ensure ReTrace-Bench data is not accidentally loaded/used in training."""
    path_str = str(path)
    if "data/retrace_bench" in path_str or "data/retrace_bench" in path_str.replace("\\", "/"):
        raise RuntimeError(
            "ReTrace-Bench is evaluation-only and must not be used for ReTrace-Learn training."
        )

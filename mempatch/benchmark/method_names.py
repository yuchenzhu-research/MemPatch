"""Canonical method and model names for the final runs."""

from __future__ import annotations

FINAL_METHODS = (
    "direct_json",
    "full_context_json",
    "summary_memory_json",
    "bm25_rag_json",
    "dense_rag_json",
    "time_aware_rag_json",
    "mempatch_noguard",
    "mempatch",
)

METHOD_ALIASES = {
    "direct_json": "direct_json",
    "frozen_direct": "direct_json",
    "full_context_json": "full_context_json",
    "full_context": "full_context_json",
    "summary_memory_json": "summary_memory_json",
    "summary_memory": "summary_memory_json",
    "bm25_rag_json": "bm25_rag_json",
    "lexical_rag": "bm25_rag_json",
    "dense_rag_json": "dense_rag_json",
    "time_aware_rag_json": "time_aware_rag_json",
    "time_aware_rag": "time_aware_rag_json",
    "mempatch_noguard": "mempatch_noguard",
    "mempatch_no_guard": "mempatch_noguard",
    "mempatch": "mempatch",
}

FINAL_MODELS = (
    "qwen3_14b",
    "mistral_nemo_12b",
    "phi4_14b",
    "deepseek_r1_qwen_14b",
    "glm4_9b",
)

FINAL_SPLITS = (
    "dev_calibration",
    "main_test_synthetic",
    "challenge_test_hard",
)

HEADLINE_SPLITS = (
    "main_test_synthetic",
    "challenge_test_hard",
)


def normalize_method_name(method: str | None) -> str:
    if not method:
        raise ValueError("method name is required")
    key = str(method).strip()
    try:
        return METHOD_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unknown MemPatch-Bench method: {method!r}") from exc


def is_final_method(method: str | None) -> bool:
    return method in FINAL_METHODS


def method_sort_key(method: str | None) -> int:
    normalized = normalize_method_name(method)
    return FINAL_METHODS.index(normalized)


def expected_cells(
    *,
    models: tuple[str, ...] = FINAL_MODELS,
    methods: tuple[str, ...] = FINAL_METHODS,
    splits: tuple[str, ...] = FINAL_SPLITS,
) -> set[tuple[str, str, str]]:
    return {(model, method, split) for model in models for method in methods for split in splits}

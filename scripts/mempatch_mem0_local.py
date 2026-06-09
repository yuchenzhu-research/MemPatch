"""Local Mem0 OSS configuration without OpenAI.

Mem0 is used only for memory ingest + semantic search. The shared MemPatch
answer model (MLX) is configured separately in ``run_mempatch_memory_baselines``.

Default stack (no API keys):

- vector store: on-disk Chroma (isolated per scenario)
- embedder: HuggingFace ``sentence-transformers`` (local)
- LLM: omitted when ``infer=False`` (recommended for fair baseline ingest)

When ``infer=True``, configure a local Ollama LLM for Mem0 fact extraction.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal

EmbedderProvider = Literal["huggingface", "ollama"]
LlmProvider = Literal["ollama"]

DEFAULT_HF_EMBED_MODEL = "multi-qa-MiniLM-L6-cos-v1"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_LLM_MODEL = "llama3.1:8b"


def build_local_mem0_config(
    *,
    embedder: EmbedderProvider = "huggingface",
    embed_model: str | None = None,
    ollama_base_url: str = DEFAULT_OLLAMA_URL,
    chroma_path: str | Path | None = None,
    collection_name: str = "mempatch",
    infer: bool = False,
    llm_provider: LlmProvider | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    """Return a Mem0 ``Memory.from_config`` dict using only local providers."""
    store_path = str(chroma_path) if chroma_path is not None else tempfile.mkdtemp(prefix="mempatch-mem0-")

    config: dict[str, Any] = {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": collection_name,
                "path": store_path,
            },
        },
    }

    if embedder == "huggingface":
        config["embedder"] = {
            "provider": "huggingface",
            "config": {
                "model": embed_model or DEFAULT_HF_EMBED_MODEL,
            },
        }
    elif embedder == "ollama":
        model = embed_model or DEFAULT_OLLAMA_EMBED_MODEL
        config["embedder"] = {
            "provider": "ollama",
            "config": {
                "model": model,
                "ollama_base_url": ollama_base_url,
                "embedding_dims": 768 if "nomic" in model else 512,
            },
        }
    else:
        raise ValueError(f"Unsupported embedder {embedder!r}")

    # Mem0 always constructs an LLM handle at init; default is OpenAI.
    # Point it at local Ollama even when infer=False (LLM unused for raw ingest).
    provider = llm_provider or "ollama"
    if provider != "ollama":
        raise ValueError("Local Mem0 config currently supports llm_provider='ollama' only.")
    config["llm"] = {
        "provider": "ollama",
        "config": {
            "model": llm_model or DEFAULT_OLLAMA_LLM_MODEL,
            "temperature": 0.0,
            "max_tokens": 2000,
            "ollama_base_url": ollama_base_url,
        },
    }

    return config


def create_local_memory(
    *,
    embedder: EmbedderProvider = "huggingface",
    embed_model: str | None = None,
    ollama_base_url: str = DEFAULT_OLLAMA_URL,
    chroma_path: str | Path | None = None,
    collection_name: str = "mempatch",
    infer: bool = False,
    llm_provider: LlmProvider | None = None,
    llm_model: str | None = None,
    mem0_config: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Instantiate Mem0 ``Memory`` from a local config dict."""
    try:
        from mem0 import Memory
    except ImportError as exc:
        raise RuntimeError(
            "Mem0 backend requires mem0ai. Install with: pip install mem0ai"
        ) from exc

    config = mem0_config or build_local_mem0_config(
        embedder=embedder,
        embed_model=embed_model,
        ollama_base_url=ollama_base_url,
        chroma_path=chroma_path,
        collection_name=collection_name,
        infer=infer,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    return Memory.from_config(config), config

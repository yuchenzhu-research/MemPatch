"""Build model-visible MemPatch views for external memory baselines.

All baselines share the same downstream answer step: ``build_prompt`` from
``benchmark.model_runner`` on a sanitized ``public_scenario_view`` whose
``public_input`` is filtered per backend.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from benchmark.public_view import public_scenario_view

from scripts.memory.mempatch_mem0_local import EmbedderProvider, create_local_memory

BACKENDS = ("base", "full", "rag", "mem0")

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _overlap_score(query: str, document: str) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    doc_tokens = set(_tokenize(document))
    if not doc_tokens:
        return 0.0
    hits = sum(1 for token in query_tokens if token in doc_tokens)
    return hits / len(query_tokens)


def _query_from_view(view: dict[str, Any]) -> str:
    parts: list[str] = []
    workflow = view.get("workflow_context")
    if isinstance(workflow, str) and workflow.strip():
        parts.append(workflow.strip())
    task = view.get("black_box_task")
    if isinstance(task, dict):
        prompt = task.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            parts.append(prompt.strip())
    return "\n".join(parts)


def _sorted_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [ev for ev in events if isinstance(ev, dict)],
        key=lambda ev: (ev.get("timestamp_order") or 0, str(ev.get("event_id") or "")),
    )


def select_rag_events(
    events: list[dict[str, Any]],
    *,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Return top-k events by lexical overlap with the scenario query."""
    top_k = max(top_k, 1)
    ranked = sorted(
        _sorted_events(events),
        key=lambda ev: (
            -_overlap_score(query, str(ev.get("text") or "")),
            ev.get("timestamp_order") or 0,
        ),
    )
    if not ranked:
        return []
    positive = [ev for ev in ranked if _overlap_score(query, str(ev.get("text") or "")) > 0]
    if positive:
        return positive[:top_k]
    return ranked[:top_k]


def apply_public_input_filter(
    view: dict[str, Any],
    *,
    backend: str,
    rag_top_k: int = 5,
    mem0_top_k: int = 5,
    mem0_infer: bool = False,
    mem0_config: dict[str, Any] | None = None,
    mem0_embedder: EmbedderProvider = "huggingface",
    mem0_embed_model: str | None = None,
    mem0_ollama_base_url: str | None = None,
    mem0_llm_model: str | None = None,
) -> dict[str, Any]:
    """Return a copy of *view* with ``public_input`` rewritten for *backend*."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; expected one of {BACKENDS}")

    filtered = copy.deepcopy(view)
    public_input = filtered.setdefault("public_input", {})
    events = public_input.get("event_trace") or []
    memories = public_input.get("initial_memory") or []
    if not isinstance(events, list):
        events = []
    if not isinstance(memories, list):
        memories = []

    if backend == "base":
        public_input["event_trace"] = []
        public_input["initial_memory"] = []
        filtered["memory_baseline"] = {
            "backend": backend,
            "note": "Task-only prompt without historical memory or evidence.",
        }
        return filtered

    if backend == "full":
        filtered["memory_baseline"] = {
            "backend": backend,
            "event_count": len(events),
            "memory_count": len(memories),
        }
        return filtered

    query = _query_from_view(filtered)

    if backend == "rag":
        selected = select_rag_events(events, query=query, top_k=rag_top_k)
        public_input["event_trace"] = selected
        filtered["memory_baseline"] = {
            "backend": backend,
            "rag_top_k": rag_top_k,
            "selected_event_ids": [ev.get("event_id") for ev in selected],
            "query_chars": len(query),
        }
        return filtered

    # mem0
    retrieved = retrieve_mem0_context(
        scenario_id=str(filtered.get("scenario_id") or "unknown"),
        events=_sorted_events(events),
        memories=[m for m in memories if isinstance(m, dict)],
        query=query,
        top_k=mem0_top_k,
        infer=mem0_infer,
        mem0_config=mem0_config,
        mem0_embedder=mem0_embedder,
        mem0_embed_model=mem0_embed_model,
        mem0_ollama_base_url=mem0_ollama_base_url,
        mem0_llm_model=mem0_llm_model,
    )
    public_input["event_trace"] = retrieved["event_trace"]
    public_input["initial_memory"] = retrieved["initial_memory"]
    filtered["memory_baseline"] = retrieved["memory_baseline"]
    return filtered


def build_baseline_view(
    scenario: dict[str, Any],
    *,
    backend: str,
    rag_top_k: int = 5,
    mem0_top_k: int = 5,
    mem0_infer: bool = False,
    mem0_config: dict[str, Any] | None = None,
    mem0_embedder: EmbedderProvider = "huggingface",
    mem0_embed_model: str | None = None,
    mem0_ollama_base_url: str | None = None,
    mem0_llm_model: str | None = None,
) -> dict[str, Any]:
    """Sanitize a scenario and apply a memory-baseline context policy."""
    view = public_scenario_view(scenario)
    return apply_public_input_filter(
        view,
        backend=backend,
        rag_top_k=rag_top_k,
        mem0_top_k=mem0_top_k,
        mem0_infer=mem0_infer,
        mem0_config=mem0_config,
        mem0_embedder=mem0_embedder,
        mem0_embed_model=mem0_embed_model,
        mem0_ollama_base_url=mem0_ollama_base_url,
        mem0_llm_model=mem0_llm_model,
    )


def retrieve_mem0_context(
    *,
    scenario_id: str,
    events: list[dict[str, Any]],
    memories: list[dict[str, Any]],
    query: str,
    top_k: int,
    infer: bool = False,
    mem0_config: dict[str, Any] | None = None,
    mem0_embedder: EmbedderProvider = "huggingface",
    mem0_embed_model: str | None = None,
    mem0_ollama_base_url: str | None = None,
    mem0_llm_model: str | None = None,
) -> dict[str, Any]:
    """Ingest trace events into Mem0 OSS and return retrieved context."""
    import tempfile

    from scripts.memory.mempatch_mem0_local import DEFAULT_OLLAMA_URL

    with tempfile.TemporaryDirectory(prefix=f"mempatch-mem0-{scenario_id}-") as chroma_dir:
        memory, active_config = create_local_memory(
            embedder=mem0_embedder,
            embed_model=mem0_embed_model,
            ollama_base_url=mem0_ollama_base_url or DEFAULT_OLLAMA_URL,
            chroma_path=chroma_dir,
            collection_name=f"mempatch-{scenario_id}",
            infer=infer,
            llm_model=mem0_llm_model,
            mem0_config=mem0_config,
        )
        return _run_mem0_retrieval(
            memory=memory,
            active_config=active_config,
            scenario_id=scenario_id,
            events=events,
            memories=memories,
            query=query,
            top_k=top_k,
            infer=infer,
            mem0_embedder=mem0_embedder,
            mem0_embed_model=mem0_embed_model,
        )


def _run_mem0_retrieval(
    *,
    memory: Any,
    active_config: dict[str, Any],
    scenario_id: str,
    events: list[dict[str, Any]],
    memories: list[dict[str, Any]],
    query: str,
    top_k: int,
    infer: bool,
    mem0_embedder: EmbedderProvider,
    mem0_embed_model: str | None,
) -> dict[str, Any]:
    user_id = f"mempatch:{scenario_id}"

    for memory_entry in memories:
        memory.add(
            [
                {
                    "role": "user",
                    "content": (
                        f"[memory:{memory_entry.get('memory_id')}] "
                        f"{memory_entry.get('text', '')}"
                    ),
                }
            ],
            user_id=user_id,
            infer=infer,
            metadata={"kind": "initial_memory", "memory_id": memory_entry.get("memory_id")},
        )

    for event in events:
        memory.add(
            [
                {
                    "role": "user",
                    "content": f"[event:{event.get('event_id')}] {event.get('text', '')}",
                }
            ],
            user_id=user_id,
            infer=infer,
            metadata={"kind": "event", "event_id": event.get("event_id")},
        )

    search_result = memory.search(query, filters={"user_id": user_id}, limit=max(top_k, 1))
    hits = search_result.get("results") if isinstance(search_result, dict) else search_result
    if not isinstance(hits, list):
        hits = []

    synthetic_events: list[dict[str, Any]] = []
    synthetic_memories: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    for index, hit in enumerate(hits[:top_k], start=1):
        if not isinstance(hit, dict):
            continue
        text = str(hit.get("memory") or hit.get("text") or "").strip()
        if not text:
            continue
        hit_id = str(hit.get("id") or f"mem0-{index}")
        selected_ids.append(hit_id)
        metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
        if metadata.get("kind") == "initial_memory":
            synthetic_memories.append(
                {
                    "memory_id": str(metadata.get("memory_id") or hit_id),
                    "text": text,
                    "scope": "mem0-retrieved",
                }
            )
        else:
            synthetic_events.append(
                {
                    "event_id": str(metadata.get("event_id") or hit_id),
                    "timestamp_order": index,
                    "actor_role": "mem0",
                    "trust_level": "retrieved",
                    "visibility_scope": "mem0-retrieved",
                    "event_type": "retrieval",
                    "text": text,
                    "related_memory_ids": [],
                }
            )

    if not synthetic_events and hits:
        for index, hit in enumerate(hits[:top_k], start=1):
            if not isinstance(hit, dict):
                continue
            text = str(hit.get("memory") or hit.get("text") or "").strip()
            if not text:
                continue
            synthetic_events.append(
                {
                    "event_id": f"mem0-hit-{index}",
                    "timestamp_order": index,
                    "actor_role": "mem0",
                    "trust_level": "retrieved",
                    "visibility_scope": "mem0-retrieved",
                    "event_type": "retrieval",
                    "text": text,
                    "related_memory_ids": [],
                }
            )

    return {
        "event_trace": synthetic_events,
        "initial_memory": synthetic_memories or memories,
        "memory_baseline": {
            "backend": "mem0",
            "mem0_top_k": top_k,
            "mem0_infer": infer,
            "mem0_embedder": mem0_embedder,
            "mem0_embed_model": mem0_embed_model or active_config.get("embedder", {}).get("config", {}).get("model"),
            "mem0_vector_store": active_config.get("vector_store", {}).get("provider"),
            "selected_hit_ids": selected_ids,
            "query_chars": len(query),
        },
    }

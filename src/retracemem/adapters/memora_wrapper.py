from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Dynamically locate and add Memora's agent_eval path to import BaseMemorySystem
memora_evals_path = Path(__file__).resolve().parents[4] / "reference" / "Memora" / "evals" / "agent_eval"
if str(memora_evals_path) not in sys.path and memora_evals_path.exists():
    sys.path.insert(0, str(memora_evals_path))

try:
    from base_evaluator import BaseMemorySystem
except ImportError:
    # Minimal mock fallback for testing isolation
    from abc import ABC, abstractmethod
    class BaseMemorySystem(ABC):  # type: ignore[no-redef]
        def __init__(self, user_id: str, **kwargs: Any) -> None:
            self.user_id = user_id

from retracemem.pipeline import ReTracePipeline
from retracemem.schemas import EvidenceNode


class ReTraceMemorySystem(BaseMemorySystem):
    """Memora-compatible memory system wrapper around ReTracePipeline."""

    def __init__(self, user_id: str, **kwargs: Any) -> None:
        super().__init__(user_id)
        # Use development fixture components by default if not provided
        self.pipeline = kwargs.get("pipeline")
        if self.pipeline is None:
            self.pipeline = ReTracePipeline.for_development_fixture()
        self.pipeline.reset_user(user_id)

    def get_system_name(self) -> str:
        return "retrace"

    def get_required_env_vars(self) -> List[str]:
        return []

    def initialize_client(self) -> bool:
        return True

    def process_conversation_file(self, file_path: str) -> Dict[str, Any]:
        import json
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.add_conversation_to_memory(data)

    def add_conversation_to_memory(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = conversation_data.get("session_id") or "0"
        session_date = conversation_data.get("date") or conversation_data.get("session_date") or ""
        conversation = conversation_data.get("conversation", [])

        # Formulate full text
        text_turns = []
        for turn in conversation:
            speaker = turn.get("speaker", "unknown")
            msg = turn.get("message", "")
            text_turns.append(f"{speaker}: {msg}")
        text = "\n".join(text_turns)

        evidence = EvidenceNode(
            evidence_id=f"{self.user_id}_session_{session_id}",
            session_id=str(session_id),
            timestamp=session_date or None,
            text=text,
            source_dataset="memora",
            source_pointer=f"user_{self.user_id}_session_{session_id}",
        )

        admitted_edges = self.pipeline.ingest_evidence(self.user_id, evidence)

        return {
            "status": "success",
            "session_id": session_id,
            "admitted_edges_count": len(admitted_edges),
        }

    def search_memories(
        self,
        query: str,
        limit: int = 50,
        session_date: Optional[str] = None,
        date_range: Optional[tuple] = None,
    ) -> List[Dict[str, Any]]:
        del session_date, date_range
        basis = self.pipeline.authorized_basis(self.user_id, query, limit=limit)
        results = []
        for item in basis:
            results.append({
                "memory": item.get("proposition") or item.get("text", ""),
                "score": 1.0,
                "source": "retrace",
            })
        return results

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


class MemoraAdapter:
    """Best-effort loader for Memora conversation and evaluation data."""

    def __init__(self, reference_root: str | Path = "reference/Memora") -> None:
        self.reference_root = Path(reference_root)

    def exists(self) -> bool:
        return self.reference_root.exists()

    def discover_data_roots(self) -> list[dict[str, Any]]:
        """Discover `<period>/<persona>` data roots without loading large files."""
        data_root = self._data_root()
        if data_root is None:
            return []

        roots: list[dict[str, Any]] = []
        try:
            period_dirs = sorted(path for path in data_root.iterdir() if path.is_dir())
        except OSError:
            return []

        for period_dir in period_dirs:
            try:
                persona_dirs = sorted(path for path in period_dir.iterdir() if path.is_dir())
            except OSError:
                continue
            for persona_dir in persona_dirs:
                persona_id = persona_dir.name
                conversations_dir = persona_dir / "conversations"
                question_path = persona_dir / f"evaluation_questions_{persona_id}.json"
                roots.append(
                    {
                        "period": period_dir.name,
                        "persona_id": persona_id,
                        "root": persona_dir,
                        "conversations_dir": conversations_dir,
                        "evaluation_questions_path": question_path if question_path.exists() else None,
                        "session_files": self._discover_session_files(conversations_dir),
                    }
                )
        return roots

    def discover(self) -> list[dict[str, Any]]:
        """Compatibility discovery entrypoint."""
        return self.discover_data_roots()

    def load_sessions(self, period: str, persona_id: str) -> list[dict[str, Any]]:
        root = self._persona_root(period, persona_id)
        if root is None:
            return []

        session_files = self._discover_session_files(root / "conversations")
        sessions: list[dict[str, Any]] = []
        for path in session_files:
            payload = self._read_json(path)
            if isinstance(payload, dict):
                sessions.append(self.normalize_session(payload, period=period, persona_id=persona_id, source_path=path))
        return sorted(sessions, key=self._session_sort_key)

    def load_evaluation_questions(self, period: str, persona_id: str) -> list[dict[str, Any]]:
        root = self._persona_root(period, persona_id)
        if root is None:
            return []

        path = root / f"evaluation_questions_{persona_id}.json"
        payload = self._read_json(path)
        if not isinstance(payload, dict):
            return []

        date_range = payload.get("date_range") if isinstance(payload.get("date_range"), dict) else {}
        questions_by_bucket = payload.get("questions")
        if not isinstance(questions_by_bucket, dict):
            return []

        normalized: list[dict[str, Any]] = []
        for bucket in ("remembering", "reasoning", "recommending"):
            questions = questions_by_bucket.get(bucket, [])
            if not isinstance(questions, list):
                continue
            for question in questions:
                if isinstance(question, dict):
                    normalized.append(
                        self.normalize_question(
                            question,
                            period=period,
                            persona_id=str(payload.get("persona") or persona_id),
                            task_bucket=bucket,
                            date_range=date_range,
                            source_path=path,
                        )
                    )
        return normalized

    def normalize_session(
        self,
        session: dict[str, Any],
        *,
        period: str,
        persona_id: str,
        source_path: Optional[Path] = None,
    ) -> dict[str, Any]:
        metadata = {
            "session_type": session.get("session_type"),
            "operation": session.get("operation"),
            "operation_details": session.get("operation_details", {}),
        }
        if source_path is not None:
            metadata["source_path"] = str(source_path)

        conversation = session.get("conversation")
        return {
            "session_id": session.get("session_id"),
            "period": period,
            "persona_id": str(session.get("persona") or persona_id),
            "date": session.get("date"),
            "session_type": session.get("session_type"),
            "operation": session.get("operation"),
            "operation_details": session.get("operation_details", {}),
            "conversation": conversation if isinstance(conversation, list) else [],
            "metadata": metadata,
            "raw": session,
        }

    def normalize_question(
        self,
        question: dict[str, Any],
        *,
        period: str,
        persona_id: str,
        task_bucket: str,
        date_range: dict[str, Any],
        source_path: Optional[Path] = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {"date_range": date_range}
        if source_path is not None:
            metadata["source_path"] = str(source_path)

        return {
            "persona_id": persona_id,
            "period": period,
            "task_bucket": task_bucket,
            "bucket": task_bucket,
            "question_id": question.get("question_id"),
            "question": question.get("question", ""),
            "question_date": question.get("question_date"),
            "memory_evidence": question.get("memory_evidence", {}),
            "forgetting_evidence": question.get("forgetting_evidence", {}),
            "evaluation": question.get("evaluation", {}),
            "rubric": question.get("evaluation", {}),
            "metadata": metadata,
            "raw": question,
        }

    def _data_root(self) -> Optional[Path]:
        if not self.exists():
            return None
        data_root = self.reference_root / "data"
        return data_root if data_root.exists() else self.reference_root

    def _persona_root(self, period: str, persona_id: str) -> Optional[Path]:
        data_root = self._data_root()
        if data_root is None:
            return None
        root = data_root / period / persona_id
        return root if root.exists() else None

    def _discover_session_files(self, conversations_dir: Path) -> list[Path]:
        if not conversations_dir.exists():
            return []
        try:
            return sorted(
                (path for path in conversations_dir.glob("session_*.json") if path.is_file()),
                key=self._session_file_sort_key,
            )
        except OSError:
            return []

    def _read_json(self, path: Path) -> Any:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

    def _session_file_sort_key(self, path: Path) -> tuple[int, str]:
        stem_number = path.stem.removeprefix("session_")
        return (int(stem_number) if stem_number.isdigit() else 0, path.name)

    def _session_sort_key(self, session: dict[str, Any]) -> tuple[int, str]:
        session_id = session.get("session_id")
        session_number = session_id if isinstance(session_id, int) else 0
        return (session_number, str(session.get("date") or ""))

    def load_as_records(self, period: str, persona_id: str) -> tuple[list[EvidenceRecord], list[QueryRecord]]:
        """Load Memora sessions and evaluation questions, returning typed EvidenceRecord and QueryRecord lists."""
        from retracemem.schemas import EvidenceRecord, QueryRecord

        sessions = self.load_sessions(period, persona_id)
        questions = self.load_evaluation_questions(period, persona_id)

        evidence_records: list[EvidenceRecord] = []
        for s in sessions:
            s_id = s.get("session_id")
            conv = s.get("conversation") or []
            text_turns = []
            for turn in conv:
                if isinstance(turn, dict):
                    speaker = turn.get("speaker") or "unknown"
                    msg = turn.get("message") or ""
                    text_turns.append(f"{speaker}: {msg}")
                elif isinstance(turn, str):
                    text_turns.append(turn)
                else:
                    text_turns.append(str(turn))
            text = "\n".join(text_turns)

            source_path = s["metadata"].get("source_path", "")
            evidence_id = f"{persona_id}_{period}_session_{s_id}"
            session_id = f"{persona_id}_{period}_session_{s_id}"

            evidence_records.append(
                EvidenceRecord(
                    evidence_id=evidence_id,
                    session_id=session_id,
                    timestamp=s.get("date"),
                    text=text,
                    source_dataset="memora",
                    source_pointer=f"{source_path}#session_{s_id}",
                    is_raw_source=True,
                    metadata={
                        "persona_id": s.get("persona_id"),
                        "period": s.get("period"),
                        "session_type": s.get("session_type"),
                        "operation": s.get("operation"),
                        "operation_details": s.get("operation_details"),
                        "original_session": s,
                    }
                )
            )

        query_records: list[QueryRecord] = []
        for q in questions:
            q_id = q.get("question_id")
            query_records.append(
                QueryRecord(
                    query_id=f"{persona_id}_{period}_{q_id}",
                    query_text=q.get("question", ""),
                    timestamp=q.get("question_date"),
                    metadata={
                        "persona_id": q.get("persona_id"),
                        "period": q.get("period"),
                        "task_bucket": q.get("task_bucket"),
                        "memory_evidence": q.get("memory_evidence"),
                        "forgetting_evidence": q.get("forgetting_evidence"),
                        "evaluation": q.get("evaluation"),
                        "rubric": q.get("rubric"),
                        "original_question": q,
                    }
                )
            )

        return evidence_records, query_records


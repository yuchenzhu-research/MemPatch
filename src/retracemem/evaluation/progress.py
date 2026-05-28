from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import TextIO

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional fallback
    tqdm = None


@dataclass
class ProgressSnapshot:
    phase: str
    stage: str
    scenarios_done: int
    scenarios_total: int
    queries_done: int
    queries_total: int
    semantic_invocations: int
    outbound_network_calls: int
    max_calls: int
    cache_hits: int
    cache_misses: int
    tokens_from_outbound_calls: int
    max_tokens: int
    current_id: str = ""


class ProgressReporter:
    def __init__(
        self,
        *,
        mode: str = "auto",
        every: int = 1,
        stream: TextIO | None = None,
        log_file: str | None = None,
    ) -> None:
        if mode not in {"auto", "bar", "line", "off"}:
            raise ValueError(f"Unsupported progress mode: {mode}")
        self.stream = stream or sys.stdout
        self.mode = self._resolve_mode(mode)
        self.every = max(every, 1)
        self.started_at = time.time()
        self._last_line_key: tuple[str, int] | None = None
        self._bar = None
        self._last_snapshot: ProgressSnapshot | None = None
        self._log_handle = open(log_file, "a", encoding="utf-8") if log_file else None

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def plan(self, message: str) -> None:
        self._emit(f"[PLAN] {message}")

    def phase(self, message: str) -> None:
        self._emit(f"[PHASE] {message}")

    def done(self, message: str) -> None:
        self._emit(f"[DONE] {message}")

    def update(self, snapshot: ProgressSnapshot) -> None:
        self._last_snapshot = snapshot
        if self.mode == "off":
            return
        if self.mode == "bar":
            self._update_bar(snapshot)
            return
        line_key = (snapshot.stage, snapshot.scenarios_done)
        if line_key == self._last_line_key:
            return
        if snapshot.scenarios_done % self.every != 0 and snapshot.scenarios_done != snapshot.scenarios_total:
            return
        self._last_line_key = line_key
        self._emit(self._format_snapshot(snapshot))

    @property
    def last_snapshot(self) -> ProgressSnapshot | None:
        return self._last_snapshot

    def _resolve_mode(self, mode: str) -> str:
        if mode == "auto":
            if os.environ.get("CI") or not self.stream.isatty() or tqdm is None:
                return "line"
            return "bar"
        if mode == "bar" and tqdm is None:
            return "line"
        return mode

    def _update_bar(self, snapshot: ProgressSnapshot) -> None:
        if tqdm is None:
            self._emit(self._format_snapshot(snapshot))
            return
        if self._bar is None or self._bar.total != snapshot.queries_total:
            if self._bar is not None:
                self._bar.close()
            self._bar = tqdm(total=snapshot.queries_total, file=self.stream, dynamic_ncols=True)
        self._bar.n = snapshot.queries_done
        self._bar.set_description(f"{snapshot.stage} {snapshot.current_id}".strip())
        self._bar.set_postfix_str(
            f"semantic={snapshot.semantic_invocations} outbound={snapshot.outbound_network_calls}/{snapshot.max_calls} "
            f"cache={snapshot.cache_hits}/{snapshot.cache_misses} tokens={snapshot.tokens_from_outbound_calls}/{snapshot.max_tokens}"
        )
        self._bar.refresh()
        if self._log_handle is not None:
            self._write_log(self._format_snapshot(snapshot))

    def _format_snapshot(self, snapshot: ProgressSnapshot) -> str:
        pct = 100.0 * snapshot.scenarios_done / snapshot.scenarios_total if snapshot.scenarios_total else 100.0
        elapsed = time.time() - self.started_at
        eta = "unknown"
        if snapshot.scenarios_done > 0 and snapshot.scenarios_total > snapshot.scenarios_done:
            remaining = elapsed / snapshot.scenarios_done * (snapshot.scenarios_total - snapshot.scenarios_done)
            eta = self._fmt_seconds(remaining)
        return (
            f"[{snapshot.stage}] scenarios {snapshot.scenarios_done}/{snapshot.scenarios_total} ({pct:.1f}%) | "
            f"queries {snapshot.queries_done}/{snapshot.queries_total} | semantic={snapshot.semantic_invocations} | "
            f"outbound={snapshot.outbound_network_calls}/{snapshot.max_calls} | cache={snapshot.cache_hits}/{snapshot.cache_misses} | "
            f"tokens={snapshot.tokens_from_outbound_calls}/{snapshot.max_tokens} | elapsed {self._fmt_seconds(elapsed)} | "
            f"ETA {eta} | uid={self._safe_id(snapshot.current_id)}"
        )

    def _emit(self, line: str) -> None:
        if self.mode != "off":
            print(line, file=self.stream, flush=True)
        self._write_log(line)

    def _write_log(self, line: str) -> None:
        if self._log_handle is not None:
            self._log_handle.write(line + "\n")
            self._log_handle.flush()

    @staticmethod
    def _fmt_seconds(seconds: float) -> str:
        seconds_int = max(int(seconds), 0)
        minutes, sec = divmod(seconds_int, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{sec:02d}"
        return f"{minutes:02d}:{sec:02d}"

    @staticmethod
    def _safe_id(value: str) -> str:
        clean = "".join(ch for ch in value if ch.isalnum() or ch in {"_", "-", ":"})
        return clean[:48]

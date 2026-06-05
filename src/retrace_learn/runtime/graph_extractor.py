"""Scenario View Builder — MemPatch Revision Module Step 1.

``V ← BuildScenarioRevisionView(S, M)``: turns ``scenario`` / ``event_trace``
and visible memory candidates into a structured revision view
(``evidence_nodes``, ``belief_nodes``, ``condition_nodes``,
``candidate_replacement_beliefs``, ``dependency_edges``).

Two implementations:

* :class:`RuleBasedGraphExtractor` — deterministic parser over dialogue markup
  (``@EVIDENCE`` / ``@BELIEF`` / ``@REPLACEMENT`` / ``@CONDITION`` /
  ``@REQUIRES``). Offline oracle for SFT labels and smoke tests.
* :class:`LearnedGraphExtractor` — thin wrapper around a text ``generate_fn``
  that parses a JSON revision view from the completion. Owns prompt assembly +
  fail-closed parsing so a trained checkpoint can be dropped in.

The two share the markup spec so a model trained on ``(event_trace, view)``
pairs learns to reproduce what the rule-based oracle extracts.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Protocol

from retracemem.multiagent.parser import (
    StructuredParseError,
    extract_json_object,
)

from retrace_learn.schemas import SchemaValidationError, validate_memory_graph

# Markup tokens. Each may be preceded by a speaker prefix like
# "Subagent A (planner): ". The extractor ignores everything before the token.
_EVIDENCE_RE = re.compile(r"@EVIDENCE\s+(\S+)(?:\s*\[t=([^\]]*)\])?\s*:\s*(.*)$")
_BELIEF_RE = re.compile(r"@BELIEF\s+(\S+)\s*<-\s*([^:]*):\s*(.*)$")
_REPLACEMENT_RE = re.compile(r"@REPLACEMENT\s+(\S+)\s*<-\s*([^:]*):\s*(.*)$")
_CONDITION_RE = re.compile(r"@CONDITION\s+(\S+)\s*@(\S+)\s*:\s*(.*)$")
_REQUIRES_RE = re.compile(r"@REQUIRES\s+(\S+)\s*->\s*(\S+)")


def empty_graph() -> dict[str, Any]:
    return {
        "evidence_nodes": [],
        "belief_nodes": [],
        "condition_nodes": [],
        "candidate_replacement_beliefs": [],
        "dependency_edges": [],
    }


def _parse_evidence_ids(blob: str) -> list[str]:
    return [tok.strip() for tok in blob.replace(";", ",").split(",") if tok.strip()]


class GraphExtractor(Protocol):
    extractor_version: str

    def extract(
        self,
        raw_dialogue: str,
        memory_snapshot: Any | None = None,
        *,
        subagent_roles: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...


class RuleBasedGraphExtractor:
    """Deterministic markup -> memory graph parser (the offline oracle)."""

    extractor_version = "rule_based_v1"

    def extract(
        self,
        raw_dialogue: str,
        memory_snapshot: Any | None = None,
        *,
        subagent_roles: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        graph = empty_graph()
        dep_counter = 0
        for line in raw_dialogue.splitlines():
            line = line.strip()
            if not line or "@" not in line:
                continue

            m = _EVIDENCE_RE.search(line)
            if m:
                ev_id, ts, text = m.group(1), m.group(2), m.group(3).strip()
                graph["evidence_nodes"].append(
                    {
                        "evidence_id": ev_id,
                        "timestamp": ts.strip() if ts else None,
                        "text": text,
                    }
                )
                continue

            m = _REPLACEMENT_RE.search(line)
            if m:
                bid, ev_blob, text = m.group(1), m.group(2), m.group(3).strip()
                graph["candidate_replacement_beliefs"].append(
                    {
                        "belief_id": bid,
                        "proposition": text,
                        "source_evidence_ids": _parse_evidence_ids(ev_blob),
                    }
                )
                continue

            m = _BELIEF_RE.search(line)
            if m:
                bid, ev_blob, text = m.group(1), m.group(2), m.group(3).strip()
                graph["belief_nodes"].append(
                    {
                        "belief_id": bid,
                        "proposition": text,
                        "source_evidence_ids": _parse_evidence_ids(ev_blob),
                    }
                )
                continue

            m = _CONDITION_RE.search(line)
            if m:
                cid, scope, text = m.group(1), m.group(2), m.group(3).strip()
                graph["condition_nodes"].append(
                    {"condition_id": cid, "scope_id": scope, "text": text}
                )
                continue

            m = _REQUIRES_RE.search(line)
            if m:
                bid, cid = m.group(1), m.group(2)
                graph["dependency_edges"].append(
                    {
                        "edge_id": f"dep_{dep_counter}",
                        "belief_id": bid,
                        "condition_id": cid,
                        "edge_type": "REQUIRES",
                        "inducer": self.extractor_version,
                    }
                )
                dep_counter += 1
                continue
        return graph


GENERATE_FN = Callable[[str], str]


def build_extraction_prompt(raw_dialogue: str, subagent_roles: list[str]) -> str:
    """Assemble the instruction prompt for the learned graph extractor."""
    roles = ", ".join(subagent_roles) if subagent_roles else "unknown"
    return (
        "You are the MemPatch Revision Module Scenario View Builder (Step 1). "
        "Read the scenario event_trace / dialogue and emit a single JSON revision view with keys: evidence_nodes, "
        "belief_nodes, condition_nodes, candidate_replacement_beliefs, "
        "dependency_edges.\n"
        "- evidence_nodes: {evidence_id, timestamp, text}\n"
        "- belief_nodes / candidate_replacement_beliefs: {belief_id, proposition, "
        "source_evidence_ids}\n"
        "- condition_nodes: {condition_id, scope_id, text}\n"
        "- dependency_edges: {edge_id, belief_id, condition_id, edge_type='REQUIRES'}\n"
        f"Subagent roles: {roles}\n\n"
        f"DIALOGUE:\n{raw_dialogue}\n\nJSON:"
    )


class LearnedGraphExtractor:
    """Learned extractor: prompt a text model, parse a JSON graph fail-closed.

    The ``generate_fn`` is any callable mapping prompt -> completion text (e.g. a
    vLLM/transformers wrapper around a fine-tuned 2B/4B model). Model loading and
    decoding are intentionally external; on any parse/schema failure this returns
    an empty graph rather than a malformed one.
    """

    extractor_version = "learned_v1"

    def __init__(self, generate_fn: GENERATE_FN) -> None:
        self._generate = generate_fn

    def extract(
        self,
        raw_dialogue: str,
        memory_snapshot: Any | None = None,
        *,
        subagent_roles: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prompt = build_extraction_prompt(raw_dialogue, subagent_roles or [])
        completion = self._generate(prompt)
        try:
            graph = extract_json_object(completion, require_top_level_keys={"evidence_nodes"})
        except StructuredParseError:
            return empty_graph()
        base = empty_graph()
        for key in base:
            value = graph.get(key, [])
            base[key] = value if isinstance(value, list) else []
        try:
            validate_memory_graph(base)
        except SchemaValidationError:
            return empty_graph()
        return base


class ClosedAPIGraphExtractor:
    """Closed-source API based graph extractor (TODO)."""

    extractor_version = "closed_api_v1"

    def __init__(self, client: Any) -> None:
        self.client = client

    def extract(
        self,
        raw_dialogue: str,
        memory_snapshot: Any | None = None,
        *,
        subagent_roles: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # TODO: Implement closed-source API model graph extraction
        return empty_graph()


class ReplayGraphExtractor:
    """Replay based graph extractor (TODO)."""

    extractor_version = "replay_v1"

    def __init__(self, pre_extracted_graphs: dict[str, dict[str, Any]]) -> None:
        self.pre_extracted_graphs = pre_extracted_graphs

    def extract(
        self,
        raw_dialogue: str,
        memory_snapshot: Any | None = None,
        *,
        subagent_roles: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # TODO: Replay matching dialogue graph from self.pre_extracted_graphs
        return empty_graph()

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from retracemem.schemas import Belief, EpisodicEvidence, RelationPrediction, RelationType


@dataclass(frozen=True)
class _TextInput:
    text: str
    record_id: str | None


class HeuristicRelationVerifier:
    """Deterministic local relation verifier for smoke runs."""

    _UNCERTAIN_PATTERNS = (
        "not sure",
        "unclear",
        "maybe",
        "might have changed",
        "may have changed",
        "not confirmed",
        "no details",
        "uncertain",
    )
    _CONDITION_PATTERNS = ("only if", " after ", " when ", "when cleared", "if their", "if the")
    _SUPPORT_PATTERNS = ("still", "continues to", "continue to", "remains", "same as before", "again")
    _REPLACEMENT_PATTERNS = (
        "moved to",
        "relocated to",
        "now uses",
        "changed",
        "replaced",
        "new primary",
        "primary email",
        "preferred delivery location to",
    )
    _INJURY_PATTERNS = (
        "broke",
        "broken",
        "cast",
        "sprained",
        "injury",
        "injured",
        "surgery",
        "recovering",
        "cannot",
        "can't",
        "must avoid",
        "suspended",
    )
    _MOBILITY_TOPICS = (
        "bicycle",
        "bike",
        "cycling",
        "commute",
        "drive",
        "drives",
        "tennis",
        "sport",
        "running",
        "run",
        "stairs",
        "climb",
        "racket",
    )
    _LOCATION_TOPICS = (
        "address",
        "lives",
        "live",
        "office",
        "building",
        "delivery",
        "location",
        "email",
        "contact",
    )
    _SCHEDULE_TOPICS = (
        "available",
        "availability",
        "call",
        "calls",
        "schedule",
        "routine",
        "monday",
        "mondays",
        "work",
        "works",
        "home",
        "remote",
    )

    def verify(
        self,
        new_evidence: EpisodicEvidence | str,
        candidate_belief: Belief | str,
        context: dict[str, object] | None = None,
    ) -> RelationPrediction:
        del context
        evidence = self._coerce_evidence(new_evidence)
        belief = self._coerce_belief(candidate_belief)

        if not evidence.text.strip() or not belief.text.strip():
            return self._prediction(
                RelationType.UNCERTAIN,
                evidence,
                belief,
                "Missing evidence or candidate belief; failing closed.",
                "",
                0.0,
            )

        evidence_norm = self._normalize(evidence.text)
        belief_norm = self._normalize(belief.text)

        uncertain_span = self._first_phrase(evidence_norm, self._UNCERTAIN_PATTERNS)
        if uncertain_span and self._has_topic_overlap(evidence_norm, belief_norm):
            return self._prediction(
                RelationType.UNCERTAIN,
                evidence,
                belief,
                "Evidence explicitly signals uncertainty.",
                uncertain_span,
                0.86,
            )

        condition_span = self._condition_span(evidence.text)
        if condition_span and self._has_topic_overlap(evidence_norm, belief_norm):
            return self._prediction(
                RelationType.CONDITION,
                evidence,
                belief,
                "Evidence preserves the belief under an explicit condition.",
                condition_span,
                0.78,
                condition=condition_span,
            )

        if self._is_replacement(evidence_norm, belief_norm):
            return self._prediction(
                RelationType.SUPERSEDE,
                evidence,
                belief,
                "Evidence provides a replacement for the prior belief.",
                self._replacement_span(evidence.text),
                0.82,
                target_belief_id=f"{belief.record_id or 'belief'}:replacement",
            )

        if self._is_blocker(evidence_norm, belief_norm):
            condition = self._block_condition(evidence_norm, belief_norm)
            return self._prediction(
                RelationType.BLOCK,
                evidence,
                belief,
                "Evidence defeats a prerequisite for current use of the belief.",
                self._block_span(evidence.text),
                0.8,
                condition=condition,
            )

        support_span = self._first_phrase(evidence_norm, self._SUPPORT_PATTERNS)
        if support_span and self._has_topic_overlap(evidence_norm, belief_norm):
            return self._prediction(
                RelationType.SUPPORT,
                evidence,
                belief,
                "Evidence repeats or continues the prior belief.",
                support_span,
                0.72,
            )

        if self._has_topic_overlap(evidence_norm, belief_norm):
            return self._prediction(
                RelationType.UNCERTAIN,
                evidence,
                belief,
                "Evidence overlaps the belief topic but no safe relation rule matched.",
                self._short_span(evidence.text),
                0.34,
            )

        return self._prediction(
            RelationType.NONE,
            evidence,
            belief,
            "Evidence appears unrelated to the candidate belief.",
            self._short_span(evidence.text),
            0.68,
        )

    @staticmethod
    def _coerce_evidence(value: EpisodicEvidence | str) -> _TextInput:
        if isinstance(value, EpisodicEvidence):
            return _TextInput(value.text or "", value.id)
        return _TextInput(str(value or ""), None)

    @staticmethod
    def _coerce_belief(value: Belief | str) -> _TextInput:
        if isinstance(value, Belief):
            return _TextInput(value.proposition or "", value.id)
        return _TextInput(str(value or ""), None)

    @staticmethod
    def _normalize(text: str) -> str:
        return f" {re.sub(r'[^a-z0-9@._]+', ' ', text.lower()).strip()} "

    def _is_replacement(self, evidence: str, belief: str) -> bool:
        has_replacement = self._first_phrase(evidence, self._REPLACEMENT_PATTERNS) is not None
        return has_replacement and self._contains_any(belief, self._LOCATION_TOPICS)

    def _is_blocker(self, evidence: str, belief: str) -> bool:
        has_blocker = self._first_phrase(evidence, self._INJURY_PATTERNS) is not None
        return has_blocker and self._contains_any(belief, self._MOBILITY_TOPICS)

    def _has_topic_overlap(self, evidence: str, belief: str) -> bool:
        evidence_terms = self._content_terms(evidence)
        belief_terms = self._content_terms(belief)
        if evidence_terms & belief_terms:
            return True
        return any(
            self._contains_any(evidence, group) and self._contains_any(belief, group)
            for group in (self._LOCATION_TOPICS, self._MOBILITY_TOPICS, self._SCHEDULE_TOPICS)
        )

    @staticmethod
    def _content_terms(text: str) -> set[str]:
        stopwords = {
            "the",
            "user",
            "their",
            "them",
            "this",
            "that",
            "with",
            "from",
            "will",
            "can",
            "for",
            "and",
            "are",
            "has",
            "have",
            "had",
            "was",
            "were",
            "to",
            "by",
            "in",
            "on",
            "at",
            "as",
            "be",
            "is",
        }
        return {term for term in text.split() if len(term) > 3 and term not in stopwords}

    @staticmethod
    def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
        return any(phrase in text for phrase in phrases)

    @staticmethod
    def _first_phrase(text: str, phrases: tuple[str, ...]) -> str | None:
        for phrase in phrases:
            if phrase in text:
                return phrase.strip()
        return None

    def _condition_span(self, text: str) -> str | None:
        normalized = self._normalize(text)
        if not self._first_phrase(normalized, self._CONDITION_PATTERNS):
            return None
        for marker in ("only if", "when", "after", "if"):
            match = re.search(rf"\b{re.escape(marker)}\b(.+)", text, flags=re.IGNORECASE)
            if match:
                return f"{marker} {match.group(1).strip(' .')}".strip()
        return self._short_span(text)

    def _replacement_span(self, text: str) -> str:
        return self._short_span(text)

    def _block_span(self, text: str) -> str:
        return self._short_span(text)

    def _block_condition(self, evidence: str, belief: str) -> str:
        if "license" in evidence or "drive" in belief:
            return "legal driving ability"
        if "wrist" in evidence or "racket" in evidence:
            return "wrist recovery"
        if "knee" in evidence or "stairs" in belief:
            return "stair mobility"
        if "bike" in belief or "bicycle" in belief or "cycling" in belief:
            return "cycling ability"
        if "run" in belief or "sport" in belief or "tennis" in belief:
            return "physical recovery"
        return "current prerequisite"

    @staticmethod
    def _short_span(text: str) -> str:
        return text.strip()[:160]

    @staticmethod
    def _prediction(
        relation: RelationType,
        evidence: _TextInput,
        belief: _TextInput,
        rationale: str,
        span: str,
        confidence: float,
        **extra: Any,
    ) -> RelationPrediction:
        return RelationPrediction(
            relation=relation,
            evidence_id=evidence.record_id,
            belief_id=belief.record_id,
            rationale=rationale,
            span=span,
            confidence=confidence,
            **extra,
        )

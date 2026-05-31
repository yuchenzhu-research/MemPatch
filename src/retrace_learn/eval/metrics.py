"""Metric primitives shared by the ReTrace-Learn eval scripts (stdlib only)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, float]:
        return {"precision": self.precision, "recall": self.recall, "f1": self.f1}


def set_prf(pred: set, gold: set) -> PRF:
    """Precision/recall/F1 over two sets of hashable identifiers."""
    if not pred and not gold:
        return PRF(1.0, 1.0, 1.0)
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return PRF(precision, recall, f1)


def accuracy(correct: int, total: int) -> float:
    return 1.0 if total == 0 else correct / total


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

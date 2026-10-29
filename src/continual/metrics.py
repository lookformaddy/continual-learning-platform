"""Evaluation metrics for binary tabular models (stdlib only)."""

from __future__ import annotations

import math


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def log_loss(y_true: list[int], proba: list[list[float]]) -> float:
    if not y_true:
        return 0.0
    eps = 1e-15
    total = 0.0
    for label, probs in zip(y_true, proba):
        p = min(max(probs[1], eps), 1.0 - eps)
        total += -(math.log(p) if label == 1 else math.log(1.0 - p))
    return total / len(y_true)


def confusion(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

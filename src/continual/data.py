"""Synthetic tabular streams with an abrupt concept drift.

Generic domain: rows are feature vectors, labels are binary outcomes.
Regime A and regime B differ in the sign of a subset of feature weights,
which is exactly the kind of abrupt drift (label-concept change) that
breaks a model trained only on recent data. Nothing here is real data.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Batch:
    """One streaming batch of labelled rows."""

    features: list[list[float]]
    labels: list[int]
    batch_id: int = 0
    regime: str = "A"

    def __len__(self) -> int:
        return len(self.labels)


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class StreamConfig:
    n_features: int = 8
    batch_size: int = 64
    n_batches: int = 40
    drift_at: int = 20
    drift_features: tuple[int, ...] = (0, 1, 2)
    noise: float = 0.05
    seed: int = 42


def regime_weights(
    seed: int, n_features: int, drift_features: tuple[int, ...]
) -> tuple[list[float], list[float]]:
    """Ground-truth weights for regimes A and B (B flips some signs)."""
    rng = random.Random(seed)
    weights_a = [rng.uniform(-1.0, 1.0) for _ in range(n_features)]
    weights_b = list(weights_a)
    for i in drift_features:
        weights_b[i] = -weights_a[i]
    return weights_a, weights_b


def generate_regime_batches(
    weights: list[float],
    n_batches: int,
    batch_size: int,
    noise: float,
    row_seed: int,
    regime: str = "A",
) -> list[Batch]:
    """Fresh rows under fixed weights — held-out data for one regime."""
    rng = random.Random(row_seed)
    batches: list[Batch] = []
    for batch_id in range(n_batches):
        features, labels = [], []
        for _ in range(batch_size):
            row = [rng.gauss(0.0, 1.0) for _ in range(len(weights))]
            p = _sigmoid(sum(w * x for w, x in zip(weights, row)))
            if rng.random() < noise:
                label = 1 - (1 if rng.random() < p else 0)
            else:
                label = 1 if rng.random() < p else 0
            features.append(row)
            labels.append(label)
        batches.append(Batch(features, labels, batch_id, regime))
    return batches


def generate_stream(config: StreamConfig | None = None) -> list[Batch]:
    """Generate a synthetic stream: regime A, then regime B after drift_at."""
    cfg = config or StreamConfig()
    weights_a, weights_b = regime_weights(cfg.seed, cfg.n_features, cfg.drift_features)
    rng = random.Random(cfg.seed + 1)  # rows: separate stream from weights
    batches: list[Batch] = []
    for batch_id in range(cfg.n_batches):
        regime = "A" if batch_id < cfg.drift_at else "B"
        weights = weights_a if regime == "A" else weights_b
        features, labels = [], []
        for _ in range(cfg.batch_size):
            row = [rng.gauss(0.0, 1.0) for _ in range(cfg.n_features)]
            score = sum(w * x for w, x in zip(weights, row))
            p = _sigmoid(score)
            if rng.random() < cfg.noise:
                label = 1 - (1 if rng.random() < p else 0)
            else:
                label = 1 if rng.random() < p else 0
            features.append(row)
            labels.append(label)
        batches.append(Batch(features, labels, batch_id, regime))
    return batches


def split_phases(batches: list[Batch], drift_at: int) -> tuple[list[Batch], list[Batch]]:
    """Split a stream into pre-drift and post-drift phases."""
    return batches[:drift_at], batches[drift_at:]

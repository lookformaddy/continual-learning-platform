"""Replay buffer: the standard mitigation for catastrophic forgetting.

New batches overwrite old knowledge in an online learner. Mixing a
sample of previously seen rows back into each update keeps the model
anchored to earlier regimes. Reservoir sampling keeps a uniform sample
of everything seen so far without storing the whole stream.
"""

from __future__ import annotations

import random
from collections import Counter


class ReservoirReplayBuffer:
    """Fixed-capacity reservoir over (features, label) rows."""

    def __init__(self, capacity: int = 1000, seed: int = 42) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._rows: list[tuple[list[float], int]] = []
        self._seen = 0
        self._rng = random.Random(seed)

    def add(self, features: list[list[float]], labels: list[int]) -> None:
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length")
        for row, label in zip(features, labels):
            self._seen += 1
            if len(self._rows) < self.capacity:
                self._rows.append((list(row), label))
            else:
                j = self._rng.randrange(self._seen)
                if j < self.capacity:
                    self._rows[j] = (list(row), label)

    def sample(self, k: int) -> tuple[list[list[float]], list[int]]:
        """Uniform random sample of up to k stored rows."""
        if k <= 0 or not self._rows:
            return [], []
        chosen = self._rng.sample(self._rows, min(k, len(self._rows)))
        return [list(r) for r, _ in chosen], [y for _, y in chosen]

    def sample_balanced(self, k: int) -> tuple[list[list[float]], list[int]]:
        """Class-balanced sample: equal rows per class where possible."""
        if k <= 0 or not self._rows:
            return [], []
        by_class: dict[int, list[tuple[list[float], int]]] = {}
        for row, label in self._rows:
            by_class.setdefault(label, []).append((row, label))
        per_class = max(1, k // len(by_class))
        chosen: list[tuple[list[float], int]] = []
        for rows in by_class.values():
            chosen.extend(self._rng.sample(rows, min(per_class, len(rows))))
        self._rng.shuffle(chosen)
        return [list(r) for r, _ in chosen], [y for _, y in chosen]

    def label_distribution(self) -> dict[int, int]:
        return dict(Counter(label for _, label in self._rows))

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def seen(self) -> int:
        return self._seen

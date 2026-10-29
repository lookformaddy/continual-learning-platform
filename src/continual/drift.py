"""Concept-drift detection: two complementary screens.

1. Accuracy sliding window -- catches *label-concept* drift, where the
   input distribution barely moves but the right answer changes. It
   compares recent accuracy against a reference level and fires when
   the drop exceeds a threshold over a full window.
2. Distribution shift screen -- catches *covariate* shift: per-feature
   mean movement measured in reference standard deviations, plus a
   binned population-stability-style (PSI-like) score.

Either screen alone is noisy; the retraining policy combines them.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class DriftSignal:
    detector: str
    score: float
    threshold: float
    fired: bool
    detail: dict[str, float] = field(default_factory=dict)


class AccuracyWindowDetector:
    """Fire when sliding-window accuracy drops below reference - margin."""

    def __init__(
        self,
        window: int = 200,
        reference_accuracy: float = 0.80,
        min_drop: float = 0.10,
        min_samples: int = 50,
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.reference_accuracy = reference_accuracy
        self.min_drop = min_drop
        self.min_samples = min_samples
        self._recent: deque[bool] = deque(maxlen=window)
        self.total_seen = 0

    def update(self, correct: list[bool]) -> DriftSignal:
        self._recent.extend(correct)
        self.total_seen += len(correct)
        if len(self._recent) < self.min_samples:
            return DriftSignal("accuracy_window", 0.0, self.min_drop, False,
                               {"window_accuracy": 0.0, "n": float(len(self._recent))})
        window_acc = sum(self._recent) / len(self._recent)
        drop = self.reference_accuracy - window_acc
        return DriftSignal(
            "accuracy_window", drop, self.min_drop, drop >= self.min_drop,
            {"window_accuracy": window_acc, "n": float(len(self._recent))},
        )

    def reset_reference(self, accuracy_value: float) -> None:
        """Re-baseline after a confirmed regime change / retrain."""
        self.reference_accuracy = accuracy_value
        self._recent.clear()

    @property
    def window_accuracy(self) -> float:
        if not self._recent:
            return 0.0
        return sum(self._recent) / len(self._recent)


class DistributionShiftDetector:
    """Per-feature mean shift (in reference sigmas) + PSI-like binned score.

    Fit on reference batches, then screen each new batch. A feature is
    flagged when its mean moves by more than ``z_threshold`` reference
    standard deviations. PSI is computed on 10 equal-width bins over the
    reference range, with 0.5-count smoothing.
    """

    def __init__(self, z_threshold: float = 1.0, psi_threshold: float = 0.25) -> None:
        self.z_threshold = z_threshold
        self.psi_threshold = psi_threshold
        self._means: list[float] = []
        self._stds: list[float] = []
        self._bins: list[list[float]] = []  # per-feature bin edges
        self._ref_props: list[list[float]] = []
        self.fitted = False

    def fit(self, features: list[list[float]]) -> None:
        if not features:
            raise ValueError("cannot fit on an empty batch")
        n_features = len(features[0])
        self._means, self._stds, self._bins, self._ref_props = [], [], [], []
        for j in range(n_features):
            col = [row[j] for row in features]
            mean = sum(col) / len(col)
            var = sum((v - mean) ** 2 for v in col) / len(col)
            std = math.sqrt(var) if var > 0 else 1.0
            lo, hi = min(col), max(col)
            edges = [lo + (hi - lo) * k / 10 for k in range(11)]
            if hi == lo:
                edges = [lo - 0.5 + k * 0.1 for k in range(11)]
            counts = [0.5] * 10  # smoothing so no bin is ever zero
            for v in col:
                idx = 9
                for k in range(10):
                    if edges[k] <= v < edges[k + 1] or (k == 9 and v == edges[10]):
                        idx = k
                        break
                counts[idx] += 1
            total = sum(counts)
            self._means.append(mean)
            self._stds.append(std)
            self._bins.append(edges)
            self._ref_props.append([c / total for c in counts])
        self.fitted = True

    def screen(self, features: list[list[float]]) -> DriftSignal:
        if not self.fitted:
            raise ValueError("fit() must be called before screen()")
        if not features:
            return DriftSignal("distribution_shift", 0.0, self.z_threshold, False)
        n_features = len(self._means)
        max_z, psi_total = 0.0, 0.0
        flagged: dict[str, float] = {}
        for j in range(n_features):
            col = [row[j] for row in features]
            mean = sum(col) / len(col)
            z = abs(mean - self._means[j]) / self._stds[j]
            max_z = max(max_z, z)
            if z >= self.z_threshold:
                flagged[f"feature_{j}_z"] = z
            edges = self._bins[j]
            counts = [0.5] * 10
            for v in col:
                idx = 9
                for k in range(10):
                    if edges[k] <= v < edges[k + 1] or (k == 9 and v == edges[10]):
                        idx = k
                        break
                counts[idx] += 1
            total = sum(counts)
            psi = 0.0
            for ref_p, count in zip(self._ref_props[j], counts):
                new_p = count / total
                psi += (new_p - ref_p) * math.log(new_p / ref_p)
            psi_total += psi
        mean_psi = psi_total / n_features
        fired = max_z >= self.z_threshold or mean_psi >= self.psi_threshold
        detail = dict(flagged)
        detail["max_z"] = max_z
        detail["mean_psi"] = mean_psi
        return DriftSignal("distribution_shift", max_z, self.z_threshold, fired, detail)

"""From-scratch online learners for tabular data (no sklearn).

Both learners expose the same small interface so the pipeline and the
registry can treat them interchangeably:

    partial_fit(X, y)   -- one incremental update from a batch
    predict(X)          -- hard labels
    predict_proba(X)    -- per-row [P(y=0), P(y=1)]
    state_dict() / load_state_dict() -- serialisable snapshot state
"""

from __future__ import annotations

import math
import random
from typing import Any, Protocol


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class OnlineLearner(Protocol):
    def partial_fit(self, X: list[list[float]], y: list[int]) -> None: ...
    def predict(self, X: list[list[float]]) -> list[int]: ...
    def predict_proba(self, X: list[list[float]]) -> list[list[float]]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class OnlineLogisticRegression:
    """SGD logistic regression with L2 decay, updated one batch at a time."""

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.1,
        l2: float = 0.001,
        seed: int = 42,
    ) -> None:
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        rng = random.Random(seed)
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.l2 = l2
        self.weights = [rng.uniform(-0.05, 0.05) for _ in range(n_features)]
        self.bias = 0.0
        self.n_seen = 0

    def partial_fit(self, X: list[list[float]], y: list[int]) -> None:
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        for row, label in zip(X, y):
            if len(row) != self.n_features:
                raise ValueError("row width does not match n_features")
            p = _sigmoid(self.bias + sum(w * x for w, x in zip(self.weights, row)))
            err = p - label
            for i, x in enumerate(row):
                grad = err * x + self.l2 * self.weights[i]
                self.weights[i] -= self.learning_rate * grad
            self.bias -= self.learning_rate * err
            self.n_seen += 1

    def predict_proba(self, X: list[list[float]]) -> list[list[float]]:
        out = []
        for row in X:
            p = _sigmoid(self.bias + sum(w * x for w, x in zip(self.weights, row)))
            out.append([1.0 - p, p])
        return out

    def predict(self, X: list[list[float]]) -> list[int]:
        return [1 if p[1] >= 0.5 else 0 for p in self.predict_proba(X)]

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "logreg",
            "n_features": self.n_features,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "weights": list(self.weights),
            "bias": self.bias,
            "n_seen": self.n_seen,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("kind") != "logreg":
            raise ValueError("state is not a logistic regression snapshot")
        self.n_features = int(state["n_features"])
        self.weights = [float(w) for w in state["weights"]]
        self.bias = float(state["bias"])
        self.n_seen = int(state["n_seen"])


class GaussianNaiveBayesOnline:
    """Incremental Gaussian naive Bayes via running sufficient statistics."""

    def __init__(self, n_features: int, var_smoothing: float = 1e-6) -> None:
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        self.n_features = n_features
        self.var_smoothing = var_smoothing
        self.counts = [0, 0]
        self.sums = [[0.0] * n_features for _ in range(2)]
        self.sumsq = [[0.0] * n_features for _ in range(2)]
        self.n_seen = 0

    def partial_fit(self, X: list[list[float]], y: list[int]) -> None:
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        for row, label in zip(X, y):
            if label not in (0, 1):
                raise ValueError("labels must be 0 or 1")
            self.counts[label] += 1
            for i, x in enumerate(row):
                self.sums[label][i] += x
                self.sumsq[label][i] += x * x
            self.n_seen += 1

    def _log_likelihood(self, row: list[float], label: int) -> float:
        n = self.counts[label]
        if n == 0:
            return float("-inf")
        total = sum(self.counts)
        ll = math.log(n / total)
        for i, x in enumerate(row):
            mean = self.sums[label][i] / n
            var = self.sumsq[label][i] / n - mean * mean
            var = max(var, self.var_smoothing)
            ll += -0.5 * math.log(2.0 * math.pi * var) - ((x - mean) ** 2) / (2.0 * var)
        return ll

    def predict_proba(self, X: list[list[float]]) -> list[list[float]]:
        out = []
        for row in X:
            l0, l1 = self._log_likelihood(row, 0), self._log_likelihood(row, 1)
            if l0 == float("-inf") and l1 == float("-inf"):
                out.append([0.5, 0.5])
                continue
            m = max(l0, l1)
            e0, e1 = math.exp(l0 - m), math.exp(l1 - m)
            out.append([e0 / (e0 + e1), e1 / (e0 + e1)])
        return out

    def predict(self, X: list[list[float]]) -> list[int]:
        return [1 if p[1] >= 0.5 else 0 for p in self.predict_proba(X)]

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "gaussian_nb",
            "n_features": self.n_features,
            "var_smoothing": self.var_smoothing,
            "counts": list(self.counts),
            "sums": [list(s) for s in self.sums],
            "sumsq": [list(s) for s in self.sumsq],
            "n_seen": self.n_seen,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("kind") != "gaussian_nb":
            raise ValueError("state is not a naive Bayes snapshot")
        self.n_features = int(state["n_features"])
        self.counts = [int(c) for c in state["counts"]]
        self.sums = [[float(v) for v in s] for s in state["sums"]]
        self.sumsq = [[float(v) for v in s] for s in state["sumsq"]]
        self.n_seen = int(state["n_seen"])

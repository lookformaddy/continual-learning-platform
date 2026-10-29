import math

import pytest

from continual.data import StreamConfig, generate_stream
from continual.learners import GaussianNaiveBayesOnline, OnlineLogisticRegression
from continual.metrics import accuracy, confusion, log_loss


def _separable() -> tuple[list[list[float]], list[int]]:
    rng_x = [[-2.0, -1.0], [-1.5, -0.5], [1.5, 0.5], [2.0, 1.0]]
    return rng_x, [0, 0, 1, 1]


def test_logreg_learns_separable_data():
    model = OnlineLogisticRegression(2, learning_rate=0.5)
    X, y = _separable()
    for _ in range(50):
        model.partial_fit(X, y)
    assert accuracy(y, model.predict(X)) == 1.0


def test_logreg_snapshot_roundtrip():
    model = OnlineLogisticRegression(2, learning_rate=0.5)
    X, y = _separable()
    model.partial_fit(X, y)
    state = model.state_dict()
    restored = OnlineLogisticRegression(2)
    restored.load_state_dict(state)
    assert restored.predict(X) == model.predict(X)
    assert restored.weights == model.weights


def test_logreg_rejects_bad_shapes():
    model = OnlineLogisticRegression(2)
    with pytest.raises(ValueError):
        model.partial_fit([[1.0]], [0, 1])
    with pytest.raises(ValueError):
        model.partial_fit([[1.0, 2.0, 3.0]], [0])


def test_naive_bayes_learns_and_roundtrips():
    model = GaussianNaiveBayesOnline(2)
    X, y = _separable()
    model.partial_fit(X, y)
    assert accuracy(y, model.predict(X)) == 1.0
    state = model.state_dict()
    restored = GaussianNaiveBayesOnline(2)
    restored.load_state_dict(state)
    assert restored.predict(X) == model.predict(X)


def test_naive_bayes_unseen_class_predicts_neutral():
    model = GaussianNaiveBayesOnline(2)
    probs = model.predict_proba([[0.0, 0.0]])
    assert probs[0] == [0.5, 0.5]


def test_metrics():
    assert accuracy([0, 1, 1], [0, 1, 0]) == pytest.approx(2 / 3)
    assert confusion([0, 1, 1], [0, 1, 0]) == {"tp": 1, "tn": 1, "fp": 0, "fn": 1}
    assert log_loss([1], [[0.1, 0.9]]) == pytest.approx(-math.log(0.9))
    assert accuracy([], []) == 0.0


def test_stream_has_two_regimes_and_drift():
    batches = generate_stream(StreamConfig(n_batches=6, drift_at=3, batch_size=8))
    assert [b.regime for b in batches] == ["A", "A", "A", "B", "B", "B"]
    assert all(len(b) == 8 for b in batches)

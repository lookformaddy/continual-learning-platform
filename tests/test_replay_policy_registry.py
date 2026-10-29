import pytest

from continual.drift import DriftSignal
from continual.learners import OnlineLogisticRegression
from continual.policy import RetrainPolicy
from continual.registry import ModelRegistry
from continual.replay import ReservoirReplayBuffer


def test_reservoir_respects_capacity_and_counts():
    buf = ReservoirReplayBuffer(capacity=10, seed=1)
    buf.add([[float(i)] for i in range(100)], [i % 2 for i in range(100)])
    assert len(buf) == 10
    assert buf.seen == 100
    assert sum(buf.label_distribution().values()) == 10


def test_reservoir_balanced_sample_mixes_classes():
    buf = ReservoirReplayBuffer(capacity=100, seed=2)
    buf.add([[float(i)] for i in range(40)], [0] * 20 + [1] * 20)
    _, labels = buf.sample_balanced(20)
    assert set(labels) == {0, 1}


def test_policy_gates_and_actions():
    policy = RetrainPolicy(cooldown_batches=3, min_batches_since_action=2)
    acc_signal = DriftSignal("accuracy_window", 0.3, 0.1, True)
    dist_signal = DriftSignal("distribution_shift", 2.0, 1.0, True)
    assert policy.decide([], 10, 0.8).action == "none"
    assert policy.decide([acc_signal], 1, 0.8).action == "none"  # min-batches gate
    assert policy.decide([acc_signal], 2, 0.8).action == "none"  # cooldown gate
    assert policy.decide([acc_signal], 5, 0.8).action == "retrain"
    assert policy.decide([acc_signal], 5, 0.4).action == "rollback_check"
    assert policy.decide([dist_signal], 5, 0.8).action == "replay_mix"


def test_registry_snapshot_rollback_and_restore():
    reg = ModelRegistry()
    model = OnlineLogisticRegression(2)
    model.partial_fit([[1.0, 1.0]], [1])
    snap1 = reg.snapshot(model, {"acc": 0.9}, "first")
    model.partial_fit([[-1.0, -1.0]], [0])
    snap2 = reg.snapshot(model, {"acc": 0.5}, "second")
    assert snap2.version == snap1.version + 1
    assert reg.rollback(snap1.version).version == snap1.version
    reg.restore_into(model, snap1.version)
    assert model.weights == snap1.state["weights"]
    with pytest.raises(KeyError):
        reg.get(999)
    with pytest.raises(ValueError):
        ModelRegistry().restore_into(model)

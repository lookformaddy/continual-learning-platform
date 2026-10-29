"""Streaming pipeline: predict -> detect -> update -> act -> snapshot.

Order matters and is deliberate: predictions for a batch are scored
*before* the learner sees its labels (true online evaluation, no
leakage). Drift detectors observe that pre-update accuracy and the
batch's feature distribution. Only then does the learner update, mixing
in replay rows when replay is enabled. The policy may then trigger a
replay retrain (extra passes over the buffer) or a rollback check
against the best snapshot on record.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from continual.data import Batch
from continual.drift import (
    AccuracyWindowDetector,
    DistributionShiftDetector,
    DriftSignal,
)
from continual.learners import OnlineLogisticRegression
from continual.metrics import accuracy
from continual.policy import PolicyDecision, RetrainPolicy
from continual.registry import ModelRegistry
from continual.replay import ReservoirReplayBuffer


@dataclass
class PipelineConfig:
    use_replay: bool = True
    replay_per_batch: int = 32
    snapshot_every_batches: int = 10
    reference_batches: int = 3


@dataclass
class BatchReport:
    batch_id: int
    regime: str
    accuracy: float
    signals: list[DriftSignal] = field(default_factory=list)
    action: str = "none"
    action_reason: str = ""
    snapshot_version: int | None = None
    buffer_size: int = 0


class ContinualPipeline:
    def __init__(
        self,
        learner: object | None = None,
        config: PipelineConfig | None = None,
        policy: RetrainPolicy | None = None,
        buffer: ReservoirReplayBuffer | None = None,
        n_features: int = 8,
    ) -> None:
        self.learner = learner or OnlineLogisticRegression(n_features)
        self.config = config or PipelineConfig()
        self.policy = policy or RetrainPolicy()
        self.buffer = buffer or ReservoirReplayBuffer()
        self.registry = ModelRegistry()
        self.accuracy_detector = AccuracyWindowDetector()
        self.distribution_detector = DistributionShiftDetector()
        self._batches_seen = 0
        self._batches_since_action = 10**9
        self._best_snapshot: tuple[int, float] | None = None
        self.reports: list[BatchReport] = []

    # -- internal helpers -------------------------------------------------
    def _maybe_snapshot(self, acc: float, note: str) -> int | None:
        snap = self.registry.snapshot(self.learner, {"batch_accuracy": acc}, note)
        if self._best_snapshot is None or acc >= self._best_snapshot[1]:
            self._best_snapshot = (snap.version, acc)
        return snap.version

    def _replay_retrain(self, passes: int) -> None:
        for _ in range(passes):
            X, y = self.buffer.sample(self.config.replay_per_batch)
            if X:
                self.learner.partial_fit(X, y)  # type: ignore[attr-defined]

    # -- main entry point -------------------------------------------------
    def process_batch(self, batch: Batch) -> BatchReport:
        preds = self.learner.predict(batch.features)  # type: ignore[attr-defined]
        acc = accuracy(batch.labels, preds)
        correct = [p == t for p, t in zip(preds, batch.labels)]

        signals: list[DriftSignal] = [self.accuracy_detector.update(correct)]
        if self.distribution_detector.fitted:
            signals.append(self.distribution_detector.screen(batch.features))
        elif self._batches_seen < self.config.reference_batches:
            # Fit the distribution screen on the first reference batches.
            self.distribution_detector.fit(batch.features)

        decision: PolicyDecision = self.policy.decide(
            signals, self._batches_since_action, self.accuracy_detector.window_accuracy
        )
        action, reason = decision.action, decision.reason
        snapshot_version = None

        if action == "rollback_check" and self._best_snapshot is not None:
            self.registry.restore_into(self.learner, self._best_snapshot[0])
            action = "rollback"
            reason += f"; rolled back to snapshot {self._best_snapshot[0]}"
        elif action in ("retrain", "replay_mix"):
            snapshot_version = self._maybe_snapshot(acc, f"pre-{action} batch {batch.batch_id}")
            self._replay_retrain(self.policy.replay_passes)
        self._batches_since_action = (
            0 if action not in ("none",) else self._batches_since_action + 1
        )

        # Regular online update: new batch, mixed with replay rows.
        update_X = list(batch.features)
        update_y = list(batch.labels)
        if self.config.use_replay:
            rx, ry = self.buffer.sample(self.config.replay_per_batch)
            update_X.extend(rx)
            update_y.extend(ry)
        self.learner.partial_fit(update_X, update_y)  # type: ignore[attr-defined]
        self.buffer.add(batch.features, batch.labels)

        self._batches_seen += 1
        if self._batches_seen % self.config.snapshot_every_batches == 0:
            snapshot_version = self._maybe_snapshot(acc, f"periodic batch {batch.batch_id}")

        report = BatchReport(
            batch_id=batch.batch_id,
            regime=batch.regime,
            accuracy=acc,
            signals=signals,
            action=action,
            action_reason=reason,
            snapshot_version=snapshot_version,
            buffer_size=len(self.buffer),
        )
        self.reports.append(report)
        return report

    def run(self, batches: list[Batch]) -> list[BatchReport]:
        return [self.process_batch(b) for b in batches]

    def status(self) -> dict[str, object]:
        current = self.registry.current()
        return {
            "batches_seen": self._batches_seen,
            "buffer_size": len(self.buffer),
            "snapshots": len(self.registry),
            "current_snapshot": current.version if current else None,
            "window_accuracy": round(self.accuracy_detector.window_accuracy, 4),
            "last_action": self.reports[-1].action if self.reports else "none",
        }

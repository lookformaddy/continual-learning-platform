"""Eval harness: accuracy before/after drift, with and without replay.

The headline comparison is the whole point of the platform: run the
same synthetic stream through a replay pipeline and a no-replay
pipeline, then report pre-drift accuracy, post-drift accuracy, the
recovery gap, and retained accuracy — the final model scored on fresh
rows from the OLD regime. That last metric is the forgetting test:
replay should retain old-regime accuracy that a no-replay model loses,
even though replay slightly slows adaptation to the new regime. That
stability–plasticity trade-off is the honest result, and the harness
reports both sides of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from continual.data import (
    Batch,
    StreamConfig,
    generate_regime_batches,
    generate_stream,
    regime_weights,
)
from continual.learners import OnlineLogisticRegression
from continual.metrics import accuracy
from continual.pipeline import BatchReport, ContinualPipeline, PipelineConfig


@dataclass
class PhaseResult:
    pre_drift_accuracy: float
    post_drift_accuracy: float
    retained_accuracy: float = 0.0  # final model on held-out regime-A rows
    drift_actions: list[str] = field(default_factory=list)
    snapshots: int = 0

    @property
    def recovery_gap(self) -> float:
        """Post-drift minus pre-drift accuracy (negative = forgetting)."""
        return self.post_drift_accuracy - self.pre_drift_accuracy


def _phase_accuracy(reports: list[BatchReport], regime: str) -> float:
    accs = [r.accuracy for r in reports if r.regime == regime]
    return sum(accs) / len(accs) if accs else 0.0


def run_stream(
    batches: list[Batch],
    use_replay: bool,
    n_features: int = 8,
    seed: int = 42,
    stream_config: StreamConfig | None = None,
) -> tuple[PhaseResult, ContinualPipeline]:
    pipeline = ContinualPipeline(
        learner=OnlineLogisticRegression(n_features, seed=seed),
        config=PipelineConfig(use_replay=use_replay),
    )
    reports = pipeline.run(batches)
    # Forgetting check: final model scored on FRESH rows from the old
    # regime (same ground-truth weights, unseen rows). This is the
    # metric replay is supposed to protect; post-drift accuracy alone
    # rewards fast adaptation, which replay deliberately trades off.
    retained = 0.0
    if stream_config is not None:
        weights_a, _ = regime_weights(
            stream_config.seed, stream_config.n_features, stream_config.drift_features
        )
        heldout = generate_regime_batches(
            weights_a, 5, stream_config.batch_size, stream_config.noise,
            row_seed=stream_config.seed + 999, regime="A",
        )
        scores = [
            accuracy(b.labels, pipeline.learner.predict(b.features))  # type: ignore[attr-defined]
            for b in heldout
        ]
        retained = sum(scores) / len(scores)
    result = PhaseResult(
        pre_drift_accuracy=_phase_accuracy(reports, "A"),
        post_drift_accuracy=_phase_accuracy(reports, "B"),
        retained_accuracy=retained,
        drift_actions=[r.action for r in reports if r.action != "none"],
        snapshots=len(pipeline.registry),
    )
    return result, pipeline


def run_comparison(config: StreamConfig | None = None) -> dict[str, PhaseResult]:
    """Same stream, replay on vs off. Returns results keyed by variant."""
    cfg = config or StreamConfig()
    batches = generate_stream(cfg)
    with_replay, _ = run_stream(
        batches, use_replay=True, n_features=cfg.n_features, stream_config=cfg
    )
    without_replay, _ = run_stream(
        batches, use_replay=False, n_features=cfg.n_features, stream_config=cfg
    )
    return {"with_replay": with_replay, "without_replay": without_replay}


def format_comparison(results: dict[str, PhaseResult]) -> str:
    lines = ["variant        pre-drift  post-drift  gap      retained(A)  actions/snapshots"]
    for name, r in results.items():
        lines.append(
            f"{name:<14} {r.pre_drift_accuracy:.3f}      {r.post_drift_accuracy:.3f}       "
            f"{r.recovery_gap:+.3f}    {r.retained_accuracy:.3f}      "
            f"{len(r.drift_actions)} actions, {r.snapshots} snapshots"
        )
    return "\n".join(lines)

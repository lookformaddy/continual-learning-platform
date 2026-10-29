from continual.data import StreamConfig, generate_stream
from continual.eval import run_comparison, run_stream
from continual.pipeline import ContinualPipeline, PipelineConfig


def test_pipeline_runs_stream_and_snapshots():
    batches = generate_stream(StreamConfig(n_batches=12, drift_at=6, batch_size=32))
    pipeline = ContinualPipeline(config=PipelineConfig(snapshot_every_batches=5))
    reports = pipeline.run(batches)
    assert len(reports) == 12
    assert pipeline.status()["batches_seen"] == 12
    assert len(pipeline.registry) >= 2  # periodic snapshots
    assert pipeline.status()["buffer_size"] > 0


def test_pipeline_fires_actions_around_drift():
    batches = generate_stream(StreamConfig(n_batches=40, drift_at=20, batch_size=64))
    pipeline = ContinualPipeline()
    reports = pipeline.run(batches)
    actions = [r.action for r in reports if r.action != "none"]
    # Some policy action (retrain / replay_mix / rollback) should trigger
    # once accuracy collapses after the regime change.
    assert actions, "expected at least one drift-triggered action"


def test_eval_comparison_structure_and_replay_not_worse():
    results = run_comparison(StreamConfig(n_batches=40, drift_at=20, batch_size=64))
    assert set(results) == {"with_replay", "without_replay"}
    for result in results.values():
        assert 0.0 <= result.pre_drift_accuracy <= 1.0
        assert 0.0 <= result.post_drift_accuracy <= 1.0
    # Replay must not collapse post-drift accuracy relative to no-replay
    # by more than a small tolerance on this fixed-seed stream.
    assert (
        results["with_replay"].post_drift_accuracy
        >= results["without_replay"].post_drift_accuracy - 0.10
    )
    # The forgetting test: replay retains old-regime accuracy that the
    # no-replay model loses after the drift (fixed-seed stream).
    assert (
        results["with_replay"].retained_accuracy
        >= results["without_replay"].retained_accuracy + 0.05
    )


def test_run_stream_without_replay_uses_empty_mix():
    batches = generate_stream(StreamConfig(n_batches=4, drift_at=2, batch_size=16))
    result, pipeline = run_stream(batches, use_replay=False)
    assert result.pre_drift_accuracy >= 0.0
    assert pipeline.config.use_replay is False

# Architecture

## Components

| Module | Responsibility |
|---|---|
| `continual/data.py` | Synthetic stream generator. Regime A, then regime B after `drift_at`: a subset of feature weights flips sign (abrupt label-concept drift). Generic features, no real data. |
| `continual/learners.py` | `OnlineLogisticRegression` (SGD + L2) and `GaussianNaiveBayesOnline` (running sufficient statistics). Shared interface: `partial_fit`, `predict`, `predict_proba`, `state_dict`, `load_state_dict`. |
| `continual/drift.py` | `AccuracyWindowDetector` (sliding-window accuracy vs reference) and `DistributionShiftDetector` (per-feature z-shift + PSI-like binned score). Both emit a `DriftSignal` with score, threshold, and detail. |
| `continual/replay.py` | `ReservoirReplayBuffer`: fixed-capacity reservoir sampling (uniform over all rows seen), plus class-balanced sampling. |
| `continual/policy.py` | `RetrainPolicy`: gates (cooldown, min batches since action) and severity routing — `replay_mix` / `retrain` / `rollback_check` / `none`. |
| `continual/registry.py` | `ModelRegistry`: append-only versioned snapshots, current-pointer rollback, `restore_into`. |
| `continual/pipeline.py` | `ContinualPipeline`: per-batch order — predict (score) → detect → policy → act (snapshot / replay retrain / rollback) → online update (new batch + replay mix) → buffer → periodic snapshot. |
| `continual/eval.py` | `run_stream` / `run_comparison`: same stream through replay and no-replay pipelines; pre/post-drift accuracy and recovery gap. |
| `continual/service.py` | Optional FastAPI app (`/health`, `/status`, `/ingest`, `/predict`, `/rollback/{version}`). Import-guarded: `app` is `None` without FastAPI. |
| `continual/cli.py` | `python -m continual demo | eval`. |

## Data flow (per batch)

```
batch ──► predict ──► score (pre-update accuracy)
              │            │
              │            ▼
              │      accuracy detector update + distribution screen
              │            │
              │            ▼
              │      policy decision (gated)
              │         ├─ retrain / replay_mix → snapshot, extra replay passes
              │         └─ rollback_check → restore best snapshot (if any)
              ▼
        online update: new batch + replay mix
              │
              ▼
        buffer add ──► periodic snapshot (every N batches)
```

## Diagram description

Imagine three layers. The **ingestion layer** is the stream of labelled
batches. The **control layer** (detectors + policy + registry) observes
every batch and decides whether to act; it never touches training data
directly. The **learning layer** (learner + replay buffer) is the only
place weights change. The eval harness sits outside the loop, replaying
recorded streams through differently-configured pipelines.

## Decision log

- **Predict-before-update** — honest online evaluation; see README.
- **Reference accuracy is a fixed starting point, not auto-adapting.**
  Auto-rebaselining would absorb a slow drift into the baseline. After a
  confirmed retrain, call `reset_reference()` explicitly.
- **Distribution screen fits on the first batch only.** Fitting on a
  longer reference window is a one-line change (`reference_batches` in
  `PipelineConfig` gates how many early batches are candidates); the
  first batch (64 rows) is enough for the synthetic demo and keeps the
  reference definition crisp.
- **Rollback target is the best snapshot by batch accuracy**, not the
  latest. The latest snapshot may itself be post-collapse; best-by-metric
  is the safer default, and history is preserved either way.
- **No persistence layer.** Snapshots live in memory; adding a snapshot
  store (object storage + a manifest) is the natural next step, and the
  registry's `state_dict` design already makes snapshots serialisable
  (JSON-safe: floats, ints, strings, lists).
- **No async / queueing.** The pipeline is synchronous by design for
  auditability; a production deployment would put ingestion behind a
  queue and run policy actions as background jobs.

## Failure modes handled

- Empty batches / empty streams: metrics return 0.0; detectors report
  not-fired rather than raising (except `fit`/`screen` ordering, which
  raises a clear error).
- Unknown snapshot versions: `KeyError` on `get`, HTTP 404 on the API.
- Missing FastAPI: service module degrades to `app = None`; core tests
  still pass (service test auto-skips).
- Retrain storms: cooldown + min-batch gates in the policy.
- Bad rollback: pointer-only, fully reversible from preserved history.

# continual-learning-platform

A continual / online learning platform for tabular models: models that keep
learning from streaming batches instead of being retrained from scratch on
a schedule — and that survive when the world changes underneath them.

## The enterprise problem

Batch-trained models decay silently. A scoring rule, a market regime, a
user-behaviour shift, or a relabelled data source changes the relationship
between features and outcomes, and a model trained last quarter keeps
serving stale predictions until someone notices the metrics. Worse, naively
fine-tuning on the newest data causes **catastrophic forgetting**: the
model overwrites what it knew about earlier regimes.

This platform addresses that loop end to end:

1. **Online learners** — from-scratch incremental logistic regression
   (SGD) and Gaussian naive Bayes. No sklearn: the update rules are the
   point, and they are small enough to audit.
2. **Concept-drift detection** — two complementary screens:
   - an accuracy sliding window (catches label-concept drift, where inputs
     look the same but the right answer changes);
   - a distribution shift screen (per-feature mean shift in reference
     sigmas + a PSI-like binned score; catches covariate shift).
3. **Replay-buffer mitigation** — a reservoir buffer mixes previously seen
   rows back into each update, anchoring the model to earlier regimes.
4. **Drift-triggered retraining policy** — detector fires are gated by a
   cooldown and a minimum batch gap, so one noisy batch never triggers a
   retrain storm. Severity picks the action: replay-mix refresh for
   distribution-only signals, full replay retrain for accuracy collapse,
   rollback check when accuracy falls below the floor.
5. **Versioned snapshots with rollback** — every retrain and periodic
   checkpoint snapshots the full learner state. Rollback only moves a
   pointer, so a bad rollback is itself reversible.
6. **Eval harness** — the same synthetic stream, run with and without
   replay, reporting accuracy before/after drift, the recovery gap, and
   retained accuracy on fresh old-regime rows (the forgetting test).
   On the default stream replay retains old-regime accuracy (0.61 vs
   0.48 without replay) while adapting slightly more slowly to the new
   regime — the stability–plasticity trade-off, reported honestly.

## Architecture overview

```
streaming batches
      │  (predict BEFORE update: true online evaluation, no leakage)
      ▼
┌─────────────┐   ┌──────────────────┐   ┌────────────────────┐
│  Learners   │◄──│  Pipeline        │──►│  Drift detectors   │
│  (logreg /  │   │  (orchestrates   │   │  accuracy window + │
│   naive     │   │   predict→detect │   │  distribution      │
│   Bayes)    │   │   →update→act)   │   │  shift screen      │
└──────┬──────┘   └──────┬───────────┘   └─────────┬──────────┘
       │                  │                         │
       │           ┌──────▼───────┐          ┌──────▼───────┐
       │           │ Replay buffer│          │ Retrain      │
       │           │ (reservoir)  │          │ policy       │
       │           └──────┬───────┘          └──────┬───────┘
       │                  │                         │
       ▼                  ▼                         ▼
┌─────────────┐   ┌──────────────────┐   ┌────────────────────┐
│  Registry   │   │ Eval harness     │   │ FastAPI service    │
│  (snapshots │   │ (replay vs no-   │   │ (optional; health, │
│   + rollback)│   │  replay, pre/post│   │  ingest, predict,  │
└─────────────┘   │  drift accuracy) │   │  status, rollback) │
                  └──────────────────┘   └────────────────────┘
```

See [`docs/architecture.md`](docs/architecture.md) for the full design,
including the decision log.

## How to run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # stdlib-only core; FastAPI optional
PYTHONPATH=src python -m continual demo
PYTHONPATH=src python -m continual eval
PYTHONPATH=src python examples/run_demo.py
make test                              # or: PYTHONPATH=src pytest -q
```

Demo output is a per-batch trace (`acc`, triggered action, fired
detectors) followed by pipeline status. The `eval` command prints the
replay vs no-replay comparison across the drift point.

Docker:

```bash
docker build -t continual-learning-platform .
docker run --rm continual-learning-platform eval
```

## Key design decisions and trade-offs

- **Predict before update.** Scoring a batch before training on it is the
  only honest online metric; training-then-scoring would hide drift.
- **Two drift screens, not one.** Accuracy catches concept drift that
  distribution screens are blind to (identical inputs, changed labels);
  distribution catches covariate shift before accuracy degrades. Either
  alone is noisy, so the policy requires a fired detector *plus* gating.
- **Reservoir replay, not a sliding window.** A sliding window forgets
  old regimes by construction; a reservoir keeps a uniform sample of
  everything seen, which is exactly what mitigates forgetting. Cost: a
  fixed memory budget and stale rows that must be balanced against fresh
  data (see `replay_per_batch`).
- **Policy gates, not raw triggers.** Cooldown + minimum batch gap stop
  retrain storms; the rollback floor stops a collapsed model from
  retraining itself deeper into the new regime before a human looks.
- **Snapshots are full state, not weights only.** Counts and sufficient
  statistics for naive Bayes, `n_seen` for both learners — a partial
  snapshot would silently change future updates after a rollback.
- **From-scratch learners.** Deliberate: the update math is auditable and
  dependency-free. For production scale you would swap in a vectorised
  learner behind the same `partial_fit` / `predict_proba` / `state_dict`
  interface — the pipeline, policy, and registry do not care.
- **CI ships as `docs/github-actions-ci.yml`.** The GitHub App used for
  publishing lacks the Workflows permission, so the workflow activates by
  moving that file to `.github/workflows/ci.yml`.

## What it demonstrates

Online learning done as a system, not a notebook trick: honest online
evaluation, drift detection that distinguishes concept drift from
covariate shift, replay as a principled forgetting mitigation, gated
retraining policy, reversible model versioning, and an eval harness that
quantifies the benefit (accuracy before/after drift, with and without
replay, plus retained old-regime accuracy) — on fully synthetic data,
with a clean-room, dependency-light codebase organised like a real
service.

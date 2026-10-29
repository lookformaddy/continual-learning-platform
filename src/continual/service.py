"""Optional FastAPI service over a single in-memory pipeline.

The import is guarded: importing this module without FastAPI installed
leaves ``app = None`` instead of raising, so the core library and the
test suite work on a bare stdlib install. ``create_app()`` raises a
clear error when FastAPI is genuinely required.
"""

from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the skip test
    FastAPI = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment]
    _FASTAPI_AVAILABLE = False

from continual.data import Batch
from continual.pipeline import ContinualPipeline

FASTAPI_AVAILABLE = _FASTAPI_AVAILABLE


def create_app(pipeline: ContinualPipeline | None = None) -> Any:
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI is not installed; install requirements.txt first")
    pipe = pipeline or ContinualPipeline()

    class IngestRequest(BaseModel):
        features: list[list[float]]
        labels: list[int]

    class PredictRequest(BaseModel):
        features: list[list[float]]

    app = FastAPI(title="continual-learning-platform")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/status")
    def status() -> dict[str, Any]:
        return pipe.status()

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict[str, Any]:
        report = pipe.process_batch(
            Batch(req.features, req.labels, batch_id=pipe.status()["batches_seen"])  # type: ignore[arg-type]
        )
        return {
            "batch_id": report.batch_id,
            "accuracy": report.accuracy,
            "action": report.action,
            "snapshot_version": report.snapshot_version,
        }

    @app.post("/predict")
    def predict(req: PredictRequest) -> dict[str, Any]:
        return {"predictions": pipe.learner.predict(req.features)}  # type: ignore[attr-defined]

    @app.post("/rollback/{version}")
    def rollback(version: int) -> dict[str, Any]:
        try:
            snap = pipe.registry.restore_into(pipe.learner, version)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"restored_version": snap.version}

    return app


app = create_app() if _FASTAPI_AVAILABLE else None

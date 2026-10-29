import pytest

fastapi = pytest.importorskip("fastapi", reason="FastAPI not installed")
pytest.importorskip("httpx", reason="httpx not installed for TestClient")

from fastapi.testclient import TestClient

from continual.pipeline import ContinualPipeline
from continual.service import create_app


def test_service_health_ingest_predict_rollback():
    app = create_app(ContinualPipeline())
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    payload = {"features": [[1.0] * 8, [-1.0] * 8], "labels": [1, 0]}
    resp = client.post("/ingest", json=payload)
    assert resp.status_code == 200
    assert 0.0 <= resp.json()["accuracy"] <= 1.0
    pred = client.post("/predict", json={"features": [[1.0] * 8]})
    assert pred.json()["predictions"] in ([[0]], [[1]])
    assert client.post("/rollback/999").status_code == 404

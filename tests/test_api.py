from fastapi.testclient import TestClient

from app.api.main import app


def test_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert "pipeline_loaded" in response.json()


def test_predict():
    payload = {
        "usage": 200,
        "bill": 120,
        "support_calls": 3,
        "region": "east",
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "probability" in body
    assert "decision" in body
    assert "latency_ms" in body

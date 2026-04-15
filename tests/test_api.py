from fastapi.testclient import TestClient

from app.api import main as api_main
from app.api.main import app


def test_root(fitted_pipeline_path, monkeypatch):
    monkeypatch.setenv("BAYESPILOT_MODEL_PATH", str(fitted_pipeline_path))
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200


def test_health(fitted_pipeline_path, monkeypatch):
    monkeypatch.setenv("BAYESPILOT_MODEL_PATH", str(fitted_pipeline_path))
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert "pipeline_loaded" in response.json()


def test_predict(fitted_pipeline_path, monkeypatch):
    monkeypatch.setenv("BAYESPILOT_MODEL_PATH", str(fitted_pipeline_path))
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
    assert "model_name" in body
    assert "recommended_action" in body
    assert "rationale" in body
    assert "decision" in body
    assert "explanation" in body
    assert "latency_ms" in body
    assert "prediction_probability" in body or "probability" in body
    probability = body.get("prediction_probability", body.get("probability"))
    assert isinstance(probability, (int, float))
    assert 0.0 <= probability <= 1.0
    assert body["recommended_action"] in {"intervene", "do_nothing"}
    assert isinstance(body["latency_ms"], (int, float))
    assert body["latency_ms"] >= 0

    decision = body["decision"]
    assert "recommended_action" in decision
    assert "expected_value_action" in decision
    assert "expected_value_no_action" in decision
    assert "net_benefit" in decision
    assert "implied_probability_threshold" in decision
    assert "rationale" in decision
    assert isinstance(decision["expected_value_action"], (int, float))
    assert isinstance(decision["expected_value_no_action"], (int, float))
    assert isinstance(decision["net_benefit"], (int, float))
    assert decision["implied_probability_threshold"] is None or isinstance(
        decision["implied_probability_threshold"], (int, float)
    )
    assert isinstance(decision["rationale"], str)

    explanation = body["explanation"]
    assert "explanation_type" in explanation
    assert "explanation_summary" in explanation
    assert isinstance(explanation["explanation_type"], str)
    assert isinstance(explanation["explanation_summary"], str)
    if "top_global_features" in explanation:
        assert isinstance(explanation["top_global_features"], list)


def test_predict_returns_fallback_explanation_when_feature_importance_missing(
    fitted_pipeline_path, monkeypatch
):
    monkeypatch.setenv("BAYESPILOT_MODEL_PATH", str(fitted_pipeline_path))
    monkeypatch.setattr(api_main, "global_feature_importance", api_main.pd.DataFrame())
    payload = {
        "usage": 180,
        "bill": 100,
        "support_calls": 2,
        "region": "west",
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    explanation = response.json()["explanation"]
    assert explanation["explanation_type"] == "global_feature_importance_unavailable"
    assert "explanation_summary" in explanation

import joblib
import pandas as pd


def test_pipeline_artifact_exists():
    pipeline = joblib.load("models/artifacts/churn_pipeline.pkl")
    assert pipeline is not None


def test_pipeline_predict_proba():
    pipeline = joblib.load("models/artifacts/churn_pipeline.pkl")

    X = pd.DataFrame(
        [
            {
                "usage": 200,
                "bill": 120,
                "support_calls": 3,
                "region": "east",
            }
        ]
    )

    prob = pipeline.predict_proba(X)[0, 1]
    assert 0.0 <= prob <= 1.0

import joblib
import pandas as pd
from pathlib import Path


def _pipeline_path() -> Path:
    """Prefer Stage 2 deployment artifact; fall back to Stage 1 baseline."""
    root = Path(__file__).resolve().parents[1]
    deployed = root / "models" / "artifacts" / "deployed_pipeline.pkl"
    baseline = root / "models" / "artifacts" / "churn_pipeline.pkl"
    if deployed.exists():
        return deployed
    return baseline


def test_pipeline_artifact_exists():
    path = _pipeline_path()
    assert path.exists(), f"Train first: python -m training.train or python -m training.train_stage2"
    pipeline = joblib.load(path)
    assert pipeline is not None


def test_pipeline_predict_proba():
    path = _pipeline_path()
    assert path.exists(), "Missing pipeline artifact"
    pipeline = joblib.load(path)

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

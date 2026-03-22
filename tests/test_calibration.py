import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from training.calibration import calibrate_pipeline
from training.pipeline import build_pipeline


def test_calibrate_pipeline_probabilities_in_unit_interval():
    rng = np.random.default_rng(0)
    n = 40
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.choice(["x", "y"], size=n),
        }
    )
    y = (X["a"] + (X["b"] == "y").astype(float) > 0).astype(int).values
    est = LogisticRegression(max_iter=500)
    pipe = build_pipeline(["a"], ["b"], est)

    calibrated = calibrate_pipeline(pipe, X, y, method="sigmoid", cv=3)
    probs = calibrated.predict_proba(X)[:, 1]
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

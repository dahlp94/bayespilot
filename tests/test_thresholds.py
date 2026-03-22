import numpy as np

from training.thresholds import sweep_thresholds


def test_sweep_returns_expected_columns():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.15, 0.9, 0.05])

    df = sweep_thresholds(y_true, y_prob, start=0.2, end=0.8, step=0.2)

    expected = {
        "threshold",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "flagged_rate",
        "low_risk_rate",
    }
    assert expected.issubset(set(df.columns))
    assert len(df) >= 3


def test_multiple_thresholds_differ():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.6])
    df = sweep_thresholds(y_true, y_prob, start=0.3, end=0.7, step=0.1)
    assert df["threshold"].nunique() > 1

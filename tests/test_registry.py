import pytest

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from training.registry import SUPPORTED_MODELS, get_estimator


def test_logistic_returns_estimator():
    est = get_estimator(
        "logistic_regression",
        {"logistic_regression": {"max_iter": 500}},
    )
    assert isinstance(est, LogisticRegression)
    assert est.max_iter == 500


def test_random_forest_returns_estimator():
    est = get_estimator(
        "random_forest",
        {
            "random_forest": {
                "n_estimators": 50,
                "max_depth": 4,
                "random_state": 0,
            }
        },
    )
    assert isinstance(est, RandomForestClassifier)


def test_gradient_boosting_returns_estimator():
    est = get_estimator(
        "gradient_boosting",
        {"gradient_boosting": {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 2, "random_state": 0}},
    )
    assert isinstance(est, GradientBoostingClassifier)


def test_unsupported_raises():
    with pytest.raises(ValueError, match="Unsupported model"):
        get_estimator("xgboost", {})

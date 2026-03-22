"""Estimator registry for Stage 2 candidate models."""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

SUPPORTED_MODELS = frozenset(
    {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    }
)


def get_estimator(model_name: str, models_config: dict):
    """
    Build an sklearn estimator from a model name and the `models:` block of config.

    `models_config` should contain keys like `logistic_regression`, `random_forest`, etc.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name!r}. "
            f"Choose one of: {sorted(SUPPORTED_MODELS)}"
        )

    if model_name == "logistic_regression":
        cfg = models_config.get("logistic_regression", {})
        return LogisticRegression(max_iter=int(cfg.get("max_iter", 1000)))

    if model_name == "random_forest":
        cfg = models_config.get("random_forest", {})
        return RandomForestClassifier(
            n_estimators=int(cfg.get("n_estimators", 200)),
            max_depth=cfg.get("max_depth"),
            random_state=cfg.get("random_state", 42),
            n_jobs=-1,
        )

    if model_name == "gradient_boosting":
        cfg = models_config.get("gradient_boosting", {})
        return GradientBoostingClassifier(
            n_estimators=int(cfg.get("n_estimators", 150)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            max_depth=int(cfg.get("max_depth", 3)),
            random_state=int(cfg.get("random_state", 42)),
        )

    raise ValueError(f"Unhandled model: {model_name}")

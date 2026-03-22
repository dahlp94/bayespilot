"""Model comparison and deployment selection."""

from __future__ import annotations

from typing import Any

import pandas as pd


# Higher is better for interpretability (simple models preferred for BayesPilot story)
INTERPRETABILITY_SCORE = {
    "logistic_regression": 1.0,
    "random_forest": 0.6,
    "gradient_boosting": 0.5,
}


def add_interpretability(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["interpretability"] = out["model"].map(
        lambda m: INTERPRETABILITY_SCORE.get(m, 0.0)
    )
    return out


def rank_models(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by: auc (desc), f1 (desc), latency_ms (asc), interpretability (desc).
    """
    df = comparison_df.copy()
    df = df.sort_values(
        by=["auc", "f1", "interpretability", "latency_ms"],
        ascending=[False, False, False, True],
    )
    # Tie-break: lower latency after auc/f1/interpretability
    return df.reset_index(drop=True)


def select_deployment_candidate(ranked_df: pd.DataFrame) -> dict[str, Any]:
    """Pick top row and build a justification dict."""
    if ranked_df.empty:
        raise ValueError("Empty comparison dataframe")

    top = ranked_df.iloc[0]
    name = str(top["model"])
    justification = (
        f"Selected '{name}' as the deployment candidate: highest rank under "
        "criteria (AUC, F1, then lower inference latency, then interpretability). "
        f"AUC={top['auc']:.4f}, F1={top['f1']:.4f}, "
        f"latency_ms={top['latency_ms']:.3f}."
    )
    return {
        "model_name": name,
        "justification": justification,
        "metrics": {
            "auc": float(top["auc"]),
            "f1": float(top["f1"]),
            "precision": float(top["precision"]),
            "recall": float(top["recall"]),
            "accuracy": float(top["accuracy"]),
            "latency_ms": float(top["latency_ms"]),
            "interpretability": float(top.get("interpretability", 0.0)),
        },
    }

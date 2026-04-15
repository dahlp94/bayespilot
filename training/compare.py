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
    Sort models for Stage 2 selection.

    If business net benefit is available, prioritize it and break ties with
    technical performance. Otherwise, fall back to technical ranking only.
    """
    df = comparison_df.copy()
    if "max_expected_net_benefit" in df.columns:
        df = df.sort_values(
            by=["max_expected_net_benefit", "auc", "f1", "interpretability", "latency_ms"],
            ascending=[False, False, False, False, True],
        )
    else:
        df = df.sort_values(
            by=["auc", "f1", "interpretability", "latency_ms"],
            ascending=[False, False, False, True],
        )
    return df.reset_index(drop=True)


def select_deployment_candidate(ranked_df: pd.DataFrame) -> dict[str, Any]:
    """Pick top row and build a business-aware selection payload."""
    if ranked_df.empty:
        raise ValueError("Empty comparison dataframe")

    top = ranked_df.iloc[0]
    name = str(top["model"])
    has_business_cols = {
        "max_expected_net_benefit",
        "best_threshold",
    }.issubset(ranked_df.columns)

    if has_business_cols and pd.notna(top["max_expected_net_benefit"]):
        reason_for_selection = (
            f"Selected {name} because it has the highest expected net benefit "
            f"({top['max_expected_net_benefit']:.2f}) at threshold {top['best_threshold']:.2f}; "
            f"AUC ({top['auc']:.4f}) and F1 ({top['f1']:.4f}) were used as technical tie-breakers "
            "and supporting evidence."
        )
    else:
        reason_for_selection = (
            f"Selected {name} due to strongest technical ranking (AUC {top['auc']:.4f}, "
            f"F1 {top['f1']:.4f}); business net-benefit data was unavailable."
        )

    return {
        "selected_model": name,
        "best_threshold": (
            float(top["best_threshold"])
            if "best_threshold" in ranked_df.columns and pd.notna(top["best_threshold"])
            else None
        ),
        "max_expected_net_benefit": (
            float(top["max_expected_net_benefit"])
            if "max_expected_net_benefit" in ranked_df.columns
            and pd.notna(top["max_expected_net_benefit"])
            else None
        ),
        "reason_for_selection": reason_for_selection,
        "supporting_metrics": {
            "auc": float(top["auc"]),
            "f1": float(top["f1"]),
            "precision": float(top["precision"]),
            "recall": float(top["recall"]),
            "accuracy": float(top["accuracy"]),
            "latency_ms": float(top["latency_ms"]),
            "interpretability": float(top.get("interpretability", 0.0)),
        },
    }

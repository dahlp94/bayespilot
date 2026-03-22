"""Threshold sweep for binary classification scores."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sweep_thresholds(
    y_true,
    y_prob,
    start: float = 0.1,
    end: float = 0.9,
    step: float = 0.05,
) -> pd.DataFrame:
    """
    For each threshold, compute classification metrics and operational rates.

    Assumes positive class is 1; predictions are (y_prob >= threshold).astype(int).
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rows = []

    thresholds = np.arange(start, end + step / 2, step)
    for t in thresholds:
        t = float(round(t, 4))
        y_pred = (y_prob >= t).astype(int)
        n = len(y_true)
        pred_pos = np.sum(y_pred == 1)
        flagged_rate = pred_pos / n if n else 0.0
        low_risk_rate = np.sum(y_pred == 0) / n if n else 0.0

        rows.append(
            {
                "threshold": t,
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "flagged_rate": float(flagged_rate),
                "low_risk_rate": float(low_risk_rate),
            }
        )

    return pd.DataFrame(rows)

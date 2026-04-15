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
    intervention_cost: float = 50.0,
    churn_loss: float = 500.0,
    intervention_success_rate: float = 0.3,
) -> pd.DataFrame:
    """
    Sweep score thresholds and compute classification + business-facing metrics.

    This is an offline evaluation approximation for threshold comparison.
    We use observed true positives in evaluation data to estimate expected
    business impact at each threshold. Those outcomes are not available at
    inference time; they are only used here to compare threshold tradeoffs,
    not to simulate real-time decisions.

    Assumes positive class is 1 and predictions are
    (y_prob >= threshold).astype(int).
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")
    if step <= 0:
        raise ValueError("step must be > 0.")
    if start > end:
        raise ValueError("start must be <= end.")
    if intervention_cost < 0:
        raise ValueError("intervention_cost must be non-negative.")
    if churn_loss < 0:
        raise ValueError("churn_loss must be non-negative.")
    if not 0 <= intervention_success_rate <= 1:
        raise ValueError("intervention_success_rate must be between 0 and 1.")

    rows = []

    thresholds = np.arange(start, end + step / 2, step)
    for t in thresholds:
        t = float(round(t, 4))
        y_pred = (y_prob >= t).astype(int)
        n = len(y_true)
        predicted_positives = np.sum(y_pred == 1)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        flagged_rate = predicted_positives / n if n else 0.0
        low_risk_rate = np.sum(y_pred == 0) / n if n else 0.0

        # Simple business approximation per threshold:
        # target all predicted positives, pay intervention cost, and recover a
        # fraction of true churners based on intervention success rate.
        number_targeted = int(predicted_positives)
        intervention_cost_total = float(number_targeted * intervention_cost)
        expected_customers_saved = float(tp * intervention_success_rate)
        expected_loss_prevented = float(expected_customers_saved * churn_loss)
        expected_net_benefit = float(expected_loss_prevented - intervention_cost_total)

        rows.append(
            {
                "threshold": t,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "flagged_rate": float(flagged_rate),
                "low_risk_rate": float(low_risk_rate),
                "number_targeted": number_targeted,
                "intervention_cost_total": intervention_cost_total,
                "expected_customers_saved": expected_customers_saved,
                "expected_loss_prevented": expected_loss_prevented,
                "expected_net_benefit": expected_net_benefit,
            }
        )

    return pd.DataFrame(rows)

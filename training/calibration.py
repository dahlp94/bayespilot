"""Probability calibration for fitted classification pipelines."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV


def calibrate_pipeline(pipeline, X, y, method: str = "sigmoid", cv: int = 3):
    """
    Wrap a fitted or unfitted pipeline with CalibratedClassifierCV.

    If the pipeline is not yet fit, this will fit the calibrator (which fits the base
    estimator via cross-validation).
    """
    method = method.lower()
    if method not in ("sigmoid", "isotonic"):
        raise ValueError(f"Unsupported calibration method: {method!r}. Use 'sigmoid' or 'isotonic'.")

    return CalibratedClassifierCV(pipeline, method=method, cv=cv).fit(X, y)

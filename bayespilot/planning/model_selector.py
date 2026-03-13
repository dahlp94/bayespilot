from __future__ import annotations
import pandas as pd
from bayespilot.planning.model_spec import ModelType, TargetType

def infer_target_type(series: pd.Series) -> TargetType:
    unique_values = series.dropna().unique()
    if len(unique_values) == 2:
        return "binary"

    return "continuous"

def select_model_type(target_series: pd.Series) -> ModelType:
    target_type = infer_target_type(target_series)
    if target_type == "binary":
        return "bayesian_logistic_regression"

    return "bayesian_linear_regression"
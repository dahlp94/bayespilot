from __future__ import annotations

from typing import Tuple

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from bayespilot.planning.model_spec import ModelSpec


def _prepare_design_matrix(df: pd.DataFrame, target: str, predictors: list[str]) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[predictors].copy()
    X = pd.get_dummies(X, drop_first=True)
    y = df[target].copy()

    data = pd.concat([X, y], axis=1).dropna()
    X = data.drop(columns=[target])
    y = data[target]

    return X, y


def run_bayesian_inference(
    df: pd.DataFrame,
    spec: ModelSpec,
    draws: int = 1000,
    tune: int = 1000,
    random_seed: int = 42,
):
    X, y = _prepare_design_matrix(df, spec.target, spec.predictors)

    X_values = X.to_numpy()
    y_values = y.to_numpy()
    n_features = X_values.shape[1]

    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0.0, sigma=5.0)
        beta = pm.Normal("beta", mu=0.0, sigma=2.0, shape=n_features)

        linear_term = intercept + pm.math.dot(X_values, beta)

        if spec.model_type == "bayesian_linear_regression":
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            pm.Normal("likelihood", mu=linear_term, sigma=sigma, observed=y_values)

        elif spec.model_type == "bayesian_logistic_regression":
            pm.Bernoulli("likelihood", logit_p=linear_term, observed=y_values)

        else:
            raise ValueError(f"Unsupported model type: {spec.model_type}")

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            cores=1,
            target_accept=0.9,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=False,
        )

    return {
        "idata": idata,
        "feature_names": X.columns.tolist(),
        "n_observations": len(y_values),
    }


def summarize_posterior(idata, feature_names: list[str]) -> pd.DataFrame:
    summary = az.summary(idata, var_names=["intercept", "beta"], round_to=3)

    beta_rows = []
    for i, name in enumerate(feature_names):
        row_name = f"beta[{i}]"
        if row_name in summary.index:
            row = summary.loc[row_name].to_dict()
            row["feature"] = name
            beta_rows.append(row)

    result = pd.DataFrame(beta_rows)
    if not result.empty:
        cols = ["feature"] + [c for c in result.columns if c != "feature"]
        result = result[cols]

    return result
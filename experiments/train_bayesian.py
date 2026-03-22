from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.inference import run_bayesian_inference
from training.planning.model_spec import ModelSpec


def train_bayesian():
    df = pd.read_csv("datasets/churn.csv")

    spec = ModelSpec(
        target="churn",
        predictors=["usage", "bill", "support_calls"],
        model_type="bayesian_logistic_regression",
        target_type="binary",
        question="Which variables influence churn?",
    )

    results = run_bayesian_inference(df, spec)

    return results


if __name__ == "__main__":
    mlflow.set_experiment("bayespilot_bayesian")

    with mlflow.start_run():
        train_bayesian()

        mlflow.log_param("model_type", "bayesian_logistic")

        print("Bayesian model run complete")

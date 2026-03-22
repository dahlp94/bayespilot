from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from training.evaluate import compute_classification_metrics
from training.pipeline import build_pipeline


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "configs/training_config.yaml") -> None:
    os.chdir(_ROOT)
    cfg = Path(config_path)
    if not cfg.is_absolute():
        cfg = _ROOT / cfg
    config = load_config(str(cfg))

    data_path = _ROOT / config["data"]["path"]
    target = config["data"]["target"]

    test_size = config["split"]["test_size"]
    random_seed = config["split"]["random_seed"]
    stratify_enabled = config["split"]["stratify"]

    max_iter = config["model"]["max_iter"]
    pipeline_path = _ROOT / config["artifacts"]["pipeline_path"]
    experiment_name = config["mlflow"]["experiment_name"]

    df = pd.read_csv(data_path)

    X = df.drop(columns=[target])
    y = df[target]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    stratify_y = y if stratify_enabled else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_y,
    )

    pipeline = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        max_iter=max_iter,
    )

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = compute_classification_metrics(y_test, y_pred, y_prob)

        mlflow.log_param("model_type", config["model"]["type"])
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_seed", random_seed)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("numeric_features", len(numeric_features))
        mlflow.log_param("categorical_features", len(categorical_features))

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, pipeline_path)

        mlflow.log_artifact(str(pipeline_path))

        print("Training complete.")
        print(f"Saved pipeline to: {pipeline_path}")
        print("Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

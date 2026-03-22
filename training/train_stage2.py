from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.calibration import calibrate_pipeline
from training.compare import add_interpretability, rank_models, select_deployment_candidate
from training.evaluate import (
    calibration_summary,
    compute_classification_metrics,
    confusion_matrix_counts,
    plot_confusion_matrix_heatmap,
    plot_pr_curve,
    plot_roc_curve,
    pr_curve_arrays,
    roc_curve_arrays,
)
from training.pipeline import build_pipeline
from training.registry import get_estimator
from training.thresholds import sweep_thresholds


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def measure_latency_ms(model, X_sample: pd.DataFrame, n_warmup: int = 20, n_iter: int = 100) -> float:
    """Median single-row predict_proba latency in ms (API-like)."""
    for _ in range(n_warmup):
        model.predict_proba(X_sample.iloc[:1])
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        model.predict_proba(X_sample.iloc[:1])
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000.0)


def main(config_path: str = "configs/stage2_model_config.yaml") -> None:
    os.chdir(_ROOT)
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = _ROOT / cfg_path
    config = load_config(cfg_path)

    data_path = _ROOT / config["data"]["path"]
    target = config["data"]["target"]
    test_size = config["split"]["test_size"]
    random_seed = config["split"]["random_seed"]
    stratify_enabled = config["split"]["stratify"]

    model_dir = _ROOT / config["artifacts"]["model_dir"]
    deployed_path = _ROOT / config["artifacts"]["deployed_model_path"]
    reports_dir = _ROOT / config["reports"]["output_dir"]
    figures_dir = reports_dir / "figures"
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    X = df.drop(columns=[target])
    y = df[target]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    stratify_y = y if stratify_enabled else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_y,
    )

    models_block = config["models"]
    candidates = models_block["candidates"]
    cal_cfg = config.get("calibration", {})
    cal_enabled = cal_cfg.get("enabled", False)
    cal_method = cal_cfg.get("method", "sigmoid")

    thr_cfg = config.get("thresholds", {})
    sweep_cfg = thr_cfg.get("sweep", {})

    comparison_rows: list[dict] = []
    metrics_rows: list[dict] = []
    threshold_parts: list[pd.DataFrame] = []

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="stage2_comparison"):
        mlflow.log_param("n_candidates", len(candidates))
        mlflow.log_param("calibration_enabled", cal_enabled)

        for model_name in candidates:
            with mlflow.start_run(nested=True, run_name=model_name):
                estimator = get_estimator(model_name, models_block)
                pipe = build_pipeline(numeric_features, categorical_features, estimator)

                if cal_enabled:
                    final_model = calibrate_pipeline(
                        pipe, X_train, y_train, method=cal_method, cv=3
                    )
                else:
                    final_model = pipe
                    final_model.fit(X_train, y_train)

                y_pred = final_model.predict(X_test)
                y_prob = final_model.predict_proba(X_test)[:, 1]

                metrics = compute_classification_metrics(y_test, y_pred, y_prob)
                cm = confusion_matrix_counts(y_test, y_pred)
                calib = calibration_summary(y_test, y_prob)
                roc_d = roc_curve_arrays(y_test, y_prob)
                pr_d = pr_curve_arrays(y_test, y_prob)

                latency_ms = measure_latency_ms(final_model, X_test)

                row = {
                    "model": model_name,
                    **metrics,
                    "latency_ms": latency_ms,
                }
                comparison_rows.append(row)
                metrics_rows.append({**row, **cm, "mean_abs_calibration_error": calib["mean_abs_calibration_error"]})

                mlflow.log_param("model", model_name)
                mlflow.log_param("calibration", cal_enabled)
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)
                mlflow.log_metric("latency_ms", latency_ms)
                mlflow.log_metric("mean_abs_calibration_error", calib["mean_abs_calibration_error"])

                artifact_file = model_dir / f"{model_name}_pipeline.pkl"
                joblib.dump(final_model, artifact_file)
                mlflow.log_artifact(str(artifact_file))

                safe = model_name.replace(" ", "_")
                roc_path = figures_dir / f"{safe}_roc.png"
                pr_path = figures_dir / f"{safe}_pr.png"
                cm_path = figures_dir / f"{safe}_confusion_matrix.png"

                plot_roc_curve(roc_d["fpr"], roc_d["tpr"], f"ROC — {model_name}", roc_path)
                plot_pr_curve(
                    pr_d["precision"],
                    pr_d["recall"],
                    f"PR — {model_name}",
                    pr_path,
                )
                plot_confusion_matrix_heatmap(
                    y_test, y_pred, f"Confusion — {model_name}", cm_path
                )
                mlflow.log_artifact(str(roc_path))
                mlflow.log_artifact(str(pr_path))
                mlflow.log_artifact(str(cm_path))

                if sweep_cfg.get("enabled", False):
                    sweep_df = sweep_thresholds(
                        y_test,
                        y_prob,
                        start=float(sweep_cfg["start"]),
                        end=float(sweep_cfg["end"]),
                        step=float(sweep_cfg["step"]),
                    )
                    sweep_df.insert(0, "model", model_name)
                    threshold_parts.append(sweep_df)

        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df = add_interpretability(comparison_df)
        ranked = rank_models(comparison_df)

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_path = reports_dir / "metrics_summary.csv"
        metrics_df.to_csv(metrics_path, index=False)

        comparison_out = reports_dir / "model_comparison.csv"
        ranked.to_csv(comparison_out, index=False)

        if threshold_parts:
            thresh_df = pd.concat(threshold_parts, ignore_index=True)
            thresh_path = reports_dir / "threshold_summary.csv"
            thresh_df.to_csv(thresh_path, index=False)

        selection = select_deployment_candidate(ranked)
        selection["artifact_candidates"] = {
            m: str(model_dir / f"{m}_pipeline.pkl") for m in candidates
        }
        selection["deployed_path"] = str(deployed_path)

        selected_path = reports_dir / "selected_model.json"
        with open(selected_path, "w", encoding="utf-8") as f:
            json.dump(selection, f, indent=2)

        winner_name = selection["model_name"]
        winner_artifact = model_dir / f"{winner_name}_pipeline.pkl"
        shutil.copy2(winner_artifact, deployed_path)

        print("Stage 2 complete.")
        print(f"Metrics summary: {metrics_path}")
        print(f"Model comparison: {comparison_out}")
        print(f"Selected model: {winner_name}")
        print(f"Deployed artifact: {deployed_path}")


if __name__ == "__main__":
    main()

import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)


def load_data():
    return pd.read_csv("datasets/churn.csv")


def preprocess(df):
    df = df.dropna()
    X = pd.get_dummies(df.drop(columns=["churn"]), drop_first=True)
    y = df["churn"]
    return X, y


def train():
    df = load_data()
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    pred_labels = (preds > 0.5).astype(int)

    metrics = {
        "auc": roc_auc_score(y_test, preds),
        "f1": f1_score(y_test, pred_labels),
        "accuracy": accuracy_score(y_test, pred_labels),
        "precision": precision_score(y_test, pred_labels),
        "recall": recall_score(y_test, pred_labels),
    }

    return model, metrics, X.columns.tolist()


if __name__ == "__main__":
    mlflow.set_experiment("bayespilot_baseline")

    with mlflow.start_run():
        model, metrics, feature_names = train()

        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_seed", 42)
        mlflow.log_param("max_iter", 500)

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        mlflow.sklearn.log_model(model, "model")

        os.makedirs("models/artifacts", exist_ok=True)
        joblib.dump(model, "models/artifacts/logistic.pkl")
        joblib.dump(feature_names, "models/artifacts/logistic_features.pkl")

        print("Saved model and feature names.")
        print("Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
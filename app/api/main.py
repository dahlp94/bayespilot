from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import joblib
import pandas as pd

from app.monitoring.prediction_logger import log_prediction

app = FastAPI()

MODEL_PATH = "models/artifacts/logistic.pkl"
FEATURES_PATH = "models/artifacts/logistic_features.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Missing model artifact. Run: python experiments/train_baseline.py")

if not os.path.exists(FEATURES_PATH):
    raise RuntimeError("Missing feature artifact. Run: python experiments/train_baseline.py")

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)


class RequestData(BaseModel):
    usage: float
    bill: float
    support_calls: int
    region: str | None = None


@app.get("/")
def root():
    return {"message": "BayesPilot API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "n_expected_features": len(feature_names),
        "expected_features": feature_names,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": "logistic_regression_baseline",
        "artifact_path": MODEL_PATH,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
    }


@app.post("/predict")
def predict(data: RequestData):
    try:
        raw = pd.DataFrame([data.model_dump()])
        X = pd.get_dummies(raw, drop_first=True)
        X = X.reindex(columns=feature_names, fill_value=0)

        prob = float(model.predict_proba(X)[0][1])

        if prob < 0.3:
            decision = "low_risk_monitor"
        elif prob < 0.7:
            decision = "medium_risk_review"
        else:
            decision = "high_risk_escalate"

        response = {
            "input": data.model_dump(),
            "aligned_features": X.to_dict(orient="records")[0],
            "probability": prob,
            "decision": decision,
        }

        log_prediction(data.model_dump(), response)

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "received_input": data.model_dump(),
                "expected_features": feature_names,
            },
        )
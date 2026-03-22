from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from app.monitoring.prediction_logger import log_prediction
from app.services.decision import make_decision

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PIPELINE = _PROJECT_ROOT / "models" / "artifacts" / "deployed_pipeline.pkl"


def _pipeline_path() -> Path:
    override = os.environ.get("BAYESPILOT_MODEL_PATH")
    return Path(override) if override else _DEFAULT_PIPELINE


pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    path = _pipeline_path()
    pipeline = joblib.load(path)
    yield


app = FastAPI(title="BayesPilot API", lifespan=lifespan)


class PredictionRequest(BaseModel):
    usage: float
    bill: float
    support_calls: int
    region: str


@app.get("/")
def root():
    return {"message": "BayesPilot API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline_loaded": pipeline is not None,
        "artifact_path": str(_pipeline_path()),
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    start = perf_counter()

    payload = request.model_dump()
    X = pd.DataFrame([payload])

    probability = float(pipeline.predict_proba(X)[0, 1])
    decision = make_decision(probability)

    latency_ms = (perf_counter() - start) * 1000.0

    response = {
        "input": payload,
        "probability": probability,
        "decision": decision,
        "latency_ms": round(latency_ms, 3),
    }

    log_prediction(payload, response)

    return response

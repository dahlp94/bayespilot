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
_FEATURE_IMPORTANCE_PATH = _PROJECT_ROOT / "reports" / "stage2" / "feature_importance.csv"
_SELECTED_MODEL_PATH = _PROJECT_ROOT / "reports" / "stage2" / "selected_model.json"


def _pipeline_path() -> Path:
    override = os.environ.get("BAYESPILOT_MODEL_PATH")
    return Path(override) if override else _DEFAULT_PIPELINE


pipeline = None
selected_model_name: str | None = None
global_feature_importance: pd.DataFrame = pd.DataFrame(
    columns=["feature_name", "importance", "model_name"]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, selected_model_name, global_feature_importance
    path = _pipeline_path()
    pipeline = joblib.load(path)
    selected_model_name = _load_selected_model_name()
    global_feature_importance = _load_feature_importance()
    yield


def _load_selected_model_name() -> str | None:
    if not _SELECTED_MODEL_PATH.exists():
        return None
    try:
        selected = pd.read_json(_SELECTED_MODEL_PATH, typ="series")
        value = selected.get("selected_model")
        return str(value) if pd.notna(value) else None
    except Exception:
        return None


def _load_feature_importance() -> pd.DataFrame:
    if not _FEATURE_IMPORTANCE_PATH.exists():
        return pd.DataFrame(columns=["feature_name", "importance", "model_name"])
    try:
        df = pd.read_csv(_FEATURE_IMPORTANCE_PATH)
    except Exception:
        return pd.DataFrame(columns=["feature_name", "importance", "model_name"])

    required_cols = {"feature_name", "importance", "model_name"}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame(columns=["feature_name", "importance", "model_name"])
    return df


def _infer_model_name_from_pipeline() -> str:
    try:
        if hasattr(pipeline, "named_steps"):
            return pipeline.named_steps["model"].__class__.__name__
        if hasattr(pipeline, "estimator") and hasattr(pipeline.estimator, "named_steps"):
            return pipeline.estimator.named_steps["model"].__class__.__name__
    except Exception:
        pass
    return "deployed_model"


def _build_explanation(model_name: str) -> dict:
    summary = (
        "These are globally important features for the deployed model and may help "
        "interpret the prediction, but they are not case-specific attributions."
    )
    if global_feature_importance.empty:
        return {
            "explanation_type": "global_feature_importance_unavailable",
            "explanation_summary": (
                f"{summary} Global feature-importance data is currently unavailable."
            ),
            "top_global_features": [],
        }

    model_rows = global_feature_importance[
        global_feature_importance["model_name"] == model_name
    ].copy()
    if model_rows.empty:
        return {
            "explanation_type": "global_feature_importance_unavailable",
            "explanation_summary": (
                f"{summary} No stored global feature-importance rows were found for "
                f"model '{model_name}'."
            ),
            "top_global_features": [],
        }

    model_rows = model_rows.sort_values("importance", ascending=False).head(5)
    top_features = model_rows[["feature_name", "importance"]].to_dict(orient="records")
    return {
        "explanation_type": "global_feature_importance",
        "explanation_summary": summary,
        "top_global_features": top_features,
    }


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
    model_name = selected_model_name or _infer_model_name_from_pipeline()
    explanation = _build_explanation(model_name)

    latency_ms = (perf_counter() - start) * 1000.0

    response = {
        "input": payload,
        "model_name": model_name,
        "probability": probability,
        "prediction_probability": probability,
        "recommended_action": decision.get("recommended_action"),
        "rationale": decision.get("rationale"),
        "decision": decision,
        "explanation": explanation,
        "latency_ms": round(latency_ms, 3),
    }

    log_prediction(payload, response)

    return response

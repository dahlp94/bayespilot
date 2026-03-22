# BayesPilot

Reproducible ML service with **Stage 1** (single baseline pipeline) and **Stage 2** (multi-model comparison, calibration, threshold sweep, deployment selection).

## Setup

```bash
python -m venv venv_bayespilot
source venv_bayespilot/bin/activate
pip install -r requirements.txt
```

---

## Stage 1 — Baseline training

```bash
python scripts/generate_churn_data.py   # if needed
python -m training.train
```

- Config: `configs/training_config.yaml`
- Artifact: `models/artifacts/churn_pipeline.pkl`
- MLflow experiment: `BayesPilot-Stage1`

---

## Stage 2 — Experimentation & model selection

**Goal:** Train multiple candidates through one preprocessing interface, compare fairly, calibrate probabilities, sweep thresholds, and **deploy one** model as `deployed_pipeline.pkl`.

```bash
python -m training.train_stage2
```

- Config: `configs/stage2_model_config.yaml`
- Per-model artifacts: `models/artifacts/{logistic_regression,random_forest,gradient_boosting}_pipeline.pkl`
- **Production artifact:** `models/artifacts/deployed_pipeline.pkl` (copy of the selected winner)
- Reports: `reports/stage2/` (`metrics_summary.csv`, `model_comparison.csv`, `threshold_summary.csv`, `selected_model.json`, `figures/*.png`)
- MLflow experiment: `BayesPilot-Stage2`

### Selection rationale

After Stage 2, read `reports/stage2/selected_model.json` for the chosen model and written justification (AUC, F1, latency, interpretability).

---

## API

The API loads **`models/artifacts/deployed_pipeline.pkl`** (run Stage 2 after training, or set `BAYESPILOT_MODEL_PATH`).

```bash
uvicorn app.api.main:app --reload
```

Example body:

```json
{
  "usage": 200,
  "bill": 120,
  "support_calls": 3,
  "region": "east"
}
```

- **Decision** logic: `app/services/decision.py` (`make_decision`)
- **Inference:** Raw JSON → `DataFrame` → `pipeline.predict_proba` (no manual `get_dummies` in the API)

---

## Tests

```bash
pytest
```

`tests/test_pipeline.py` prefers `deployed_pipeline.pkl` if present, else `churn_pipeline.pkl`.

---

## Legacy

```bash
python experiments/old_train_baseline.py
```

Delegates to `training.train`.

## Streamlit (optional)

```bash
streamlit run app/streamlit_app.py
```

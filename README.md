# BayesPilot

BayesPilot is a production-oriented machine learning decision-support system that demonstrates the full lifecycle of an ML service, from training and evaluation to deployment, monitoring, and model selection.

The project is structured to emphasize reproducibility, consistent preprocessing, model comparison, and decision-aware inference.

---

## Overview

BayesPilot is designed as an end-to-end system with the following capabilities:

- reproducible training pipelines
- unified preprocessing for training and inference
- experiment tracking using MLflow
- multi-model comparison and selection
- probability calibration and threshold tuning
- API-based real-time inference
- logging and monitoring for predictions and latency

The system evolves across stages:

- **Stage 1**: baseline pipeline and reproducible training
- **Stage 2**: multi-model experimentation, calibration, and deployment selection

---

## Project Structure

```text
bayespilot/
├── app/
│   ├── api/
│   │   └── main.py
│   ├── monitoring/
│   │   ├── latency.py
│   │   └── prediction_logger.py
│   └── services/
│       └── decision.py
│
├── configs/
│   ├── training_config.yaml
│   └── stage2_model_config.yaml
│
├── datasets/
│   └── churn.csv
│
├── models/
│   └── artifacts/
│       ├── churn_pipeline.pkl
│       ├── logistic_regression_pipeline.pkl
│       ├── random_forest_pipeline.pkl
│       ├── gradient_boosting_pipeline.pkl
│       └── deployed_pipeline.pkl
│
├── reports/
│   └── stage2/
│       ├── metrics_summary.csv
│       ├── model_comparison.csv
│       ├── threshold_summary.csv
│       ├── selected_model.json
│       └── figures/
│
├── training/
│   ├── pipeline.py
│   ├── evaluate.py
│   ├── registry.py
│   ├── calibration.py
│   ├── thresholds.py
│   ├── compare.py
│   ├── train.py
│   └── train_stage2.py
│
├── tests/
├── scripts/
├── logs/
└── requirements.txt
````

---

## Setup

```bash
python -m venv venv_bayespilot
source venv_bayespilot/bin/activate
pip install -r requirements.txt
```

---

## Data

Synthetic churn data is generated using:

```bash
python scripts/generate_churn_data.py
```

Dataset location:

```text
datasets/churn.csv
```

Features:

* usage
* bill
* support_calls
* region (categorical)

Target:

* churn (binary)

---

## Stage 1 — Baseline Training Pipeline

Stage 1 establishes a reproducible training pipeline with unified preprocessing.

```bash
python -m training.train
```

### Key Characteristics

* preprocessing and model are combined in a single sklearn pipeline
* no duplication between training and inference
* artifact contains both preprocessing and model
* MLflow tracks parameters, metrics, and artifacts

### Outputs

* Config: `configs/training_config.yaml`
* Artifact: `models/artifacts/churn_pipeline.pkl`
* MLflow experiment: `BayesPilot-Stage1`

---

## Stage 2 — Experimentation and Model Selection

Stage 2 extends the system to support multiple candidate models under a shared training and evaluation framework.

```bash
python -m training.train_stage2
```

### Objectives

* train multiple models with identical preprocessing
* evaluate models using consistent metrics
* calibrate predicted probabilities
* perform threshold sweeps for decision tuning
* compare models and select a deployment candidate

### Candidate Models

* logistic regression
* random forest
* gradient boosting

### Outputs

* Per-model artifacts:

  ```text
  models/artifacts/{logistic_regression,random_forest,gradient_boosting}_pipeline.pkl
  ```

* Selected deployment artifact:

  ```text
  models/artifacts/deployed_pipeline.pkl
  ```

* Reports:

  ```text
  reports/stage2/
  ├── metrics_summary.csv
  ├── model_comparison.csv
  ├── threshold_summary.csv
  ├── selected_model.json
  └── figures/*.png
  ```

* MLflow experiment:

  ```text
  BayesPilot-Stage2
  ```

---

## Model Selection and Decision Criteria

The deployed model is selected based on:

* predictive performance (AUC, F1)
* probability quality after calibration
* inference latency
* interpretability considerations

The final selection and justification are recorded in:

```text
reports/stage2/selected_model.json
```

---

## API

The API serves predictions using the selected deployed model artifact.

```bash
uvicorn app.api.main:app --reload
```

### Model Loading

The API loads:

```text
models/artifacts/deployed_pipeline.pkl
```

Alternatively, override via environment variable:

```bash
export BAYESPILOT_MODEL_PATH=...
```

### Example Request

```json
{
  "usage": 200,
  "bill": 120,
  "support_calls": 3,
  "region": "east"
}
```

### Inference Flow

```text
JSON input
→ pandas DataFrame
→ sklearn pipeline (preprocessing + model)
→ predict_proba
→ decision logic
→ response
```

No manual feature engineering is performed inside the API.

---

## Decision Layer

Decision logic is defined in:

```text
app/services/decision.py
```

Mapping:

```text
probability < 0.3      → low_risk_monitor
0.3 ≤ probability < 0.7 → medium_risk_review
probability ≥ 0.7      → high_risk_escalate
```

This converts model outputs into actionable categories.

---

## Monitoring

Each prediction is logged to:

```text
logs/predictions.jsonl
```

Logged fields include:

* timestamp
* input payload
* predicted probability
* decision
* latency

This provides a basic foundation for observability and auditing.

---

## Testing

Run all tests:

```bash
pytest
```

Key test areas:

* pipeline artifact loading and inference
* decision logic correctness
* API endpoint functionality
* model registry and threshold utilities

`tests/test_pipeline.py` prioritizes `deployed_pipeline.pkl` if available.

---

## MLflow Tracking

MLflow is used for experiment tracking in both stages.

To view experiments:

```bash
mlflow ui
```

Tracked information includes:

* model parameters
* evaluation metrics
* saved artifacts
* per-model experiment runs

---

## Legacy Training Script

```bash
python experiments/old_train_baseline.py
```

This script is deprecated and redirects to the Stage 1 training pipeline.

---

## Optional: Streamlit Interface

```bash
streamlit run app/streamlit_app.py
```

Provides a simple UI for interacting with the prediction service.

---

## Design Principles

BayesPilot is built around the following principles:

* **reproducibility**: config-driven training and MLflow tracking
* **consistency**: identical preprocessing for training and inference
* **modularity**: clear separation between training, serving, and monitoring
* **comparability**: shared evaluation framework across models
* **decision-awareness**: predictions are converted into actionable outcomes
* **traceability**: all experiments and artifacts are logged and reproducible

---

## Summary

BayesPilot demonstrates how to move from a single-model pipeline to a structured ML system that supports:

* controlled experimentation
* model selection based on evidence
* consistent deployment artifacts
* real-time inference with decision logic
* monitoring and logging for operational visibility

This progression reflects a realistic transition from prototype ML workflows to production-oriented systems.
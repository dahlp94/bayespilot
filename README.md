# BayesPilot — Stage 1

## Target layout

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
├── configs/
│   └── training_config.yaml
├── datasets/
│   └── churn.csv
├── models/
│   └── artifacts/
│       └── churn_pipeline.pkl   # produced by training
├── training/
│   ├── pipeline.py
│   ├── train.py
│   └── evaluate.py
├── scripts/
│   └── generate_churn_data.py
├── tests/
│   ├── test_pipeline.py
│   ├── test_decision.py
│   └── test_api.py
├── experiments/
│   └── old_train_baseline.py
├── logs/
├── requirements.txt
└── README.md
```

Optional (Bayesian / UI): `app/streamlit_app.py`, `app/analysis/`, `training/inference.py`, `training/planning/`, `experiments/train_bayesian.py`.

## Setup

```bash
python -m venv venv_bayespilot
source venv_bayespilot/bin/activate
pip install -r requirements.txt
```

## Stage 1 end-to-end

Run from the **project root** (so paths like `datasets/churn.csv` resolve).

### 1. Generate data

```bash
python scripts/generate_churn_data.py
```

### 2. Train pipeline

```bash
python -m training.train
```

Writes `models/artifacts/churn_pipeline.pkl` (path from `configs/training_config.yaml`).  
MLflow experiment: `BayesPilot-Stage1`.

### 3. Tests

```bash
pytest
```

`tests/test_pipeline.py` expects the artifact from step 2.

### 4. API

```bash
uvicorn app.api.main:app --reload
```

### 5. Example prediction body

```json
{
  "usage": 200,
  "bill": 120,
  "support_calls": 3,
  "region": "east"
}
```

Response includes `probability`, `decision`, and `latency_ms`.

## Configuration

Single source of truth: `configs/training_config.yaml` (data path, split, `max_iter`, thresholds for future use, artifact path, MLflow experiment name).

## API notes

- Loads the **full sklearn pipeline** (preprocessing + model); no manual `get_dummies` or column alignment in the API.
- **Decision** logic lives in `app/services/decision.py` (`make_decision`).
- Startup uses FastAPI **lifespan** (not deprecated `on_event`) so `TestClient` loads the artifact reliably in tests.

## Legacy

```bash
python experiments/old_train_baseline.py
```

Delegates to `training.train` (deprecated wrapper; prefer `python -m training.train`).

## Streamlit (optional)

```bash
streamlit run app/streamlit_app.py
```

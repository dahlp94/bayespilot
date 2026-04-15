# BayesPilot Runbook

This runbook explains how to set up BayesPilot and run it end-to-end from scratch.

## Environment Setup

From the project root:

```bash
python -m venv venv_bayespilot
source venv_bayespilot/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verify Python is using the virtual environment:

```bash
which python
python --version
```

## Data Setup

BayesPilot expects `datasets/churn.csv`.

- If the dataset already exists, you can reuse it.
- If not, generate a fresh synthetic dataset:

```bash
python scripts/generate_churn_data.py
```

Expected output file:

- `datasets/churn.csv`

## Stage 1 Training

Run baseline training:

```bash
python -m training.train
```

Expected artifact:

- `models/artifacts/churn_pipeline.pkl`

This stage trains the baseline pipeline and writes one serialized model artifact.

## Stage 2 Training

Run model comparison, selection, and deployment promotion:

```bash
python -m training.train_stage2
```

Expected outputs:

- reports and evaluation outputs in `reports/stage2/` (for example, comparison and selection files)
- deployment artifact in `models/artifacts/deployed_pipeline.pkl`

Notes:

- Stage 2 trains candidate models, compares them, and promotes the selected model as the deployed artifact.
- The API uses `models/artifacts/deployed_pipeline.pkl` by default.

## Running the API

Start the FastAPI service:

```bash
uvicorn app.api.main:app --reload
```

API will be available at:

- `http://127.0.0.1:8000`

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction request example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "usage": 200,
    "bill": 120,
    "support_calls": 3,
    "region": "east"
  }'
```

Example response shape (high level):

```json
{
  "model_name": "string",
  "prediction_probability": 0.0,
  "recommended_action": "intervene or do_nothing",
  "rationale": "string",
  "decision": {
    "recommended_action": "string",
    "expected_value_action": 0.0,
    "expected_value_no_action": 0.0,
    "net_benefit": 0.0,
    "implied_probability_threshold": 0.0,
    "rationale": "string"
  },
  "explanation": {
    "explanation_type": "string",
    "explanation_summary": "string",
    "top_global_features": []
  },
  "latency_ms": 0.0
}
```

## Running Tests

Run all tests:

```bash
pytest
```

Run decision tests only:

```bash
pytest tests/test_decision.py
```

Run API tests only:

```bash
pytest tests/test_api.py
```

## Monitoring and Drift Checks

Prediction logs are written to:

- `logs/predictions.jsonl`

Run the lightweight drift-style monitoring script:

```bash
python scripts/check_prediction_drift.py
```

Optional flags:

```bash
python scripts/check_prediction_drift.py --log-path logs/predictions.jsonl --min-volume 20 --max-avg-latency-ms 50
```

## Troubleshooting

- Missing model artifacts
  - Run Stage 1 and Stage 2 again.
  - Confirm `models/artifacts/deployed_pipeline.pkl` exists before starting the API.

- Import or module errors
  - Make sure virtual environment is activated.
  - Reinstall dependencies with `pip install -r requirements.txt`.
  - Run commands from the project root.

- Empty or missing logs
  - Logs are created when `/predict` is called.
  - Send at least one prediction request before running drift checks.
  - Confirm `logs/` is writable.

- Failed tests
  - Ensure dependencies are installed in the active environment.
  - Re-run dataset generation and training if artifacts are stale or missing.
  - Start with `pytest tests/test_decision.py` and `pytest tests/test_api.py` to isolate issues.

- API starts but `/predict` fails
  - Verify deployed artifact path is valid: `models/artifacts/deployed_pipeline.pkl`.
  - Check `/health` output for `pipeline_loaded`.

## Summary

BayesPilot is reproducible with a simple flow:

1. set up environment
2. generate data
3. run Stage 1 and Stage 2 training
4. serve the API
5. run tests
6. check logs and drift script

Following this runbook should let a new engineer run the full system reliably from scratch.

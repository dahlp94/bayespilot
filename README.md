# BayesPilot — Decision-Aware ML System for Customer Retention

BayesPilot is a production-style machine learning system for churn risk estimation and retention decision support.  
It is designed to show how an ML model moves from training to operational use, where predictions are translated into actions, logged, and evaluated against business context.

## Business Problem

Customer churn prediction is useful only when it informs retention action.  
In practice, teams must decide who to contact, when to intervene, and how to prioritize limited retention capacity.

BayesPilot frames this as a decision-support problem:
- estimate churn probability for each customer
- map probability to a retention action tier
- keep the pipeline reproducible and measurable
- monitor serving behavior in production-style API flow

## Stage 1 (Baseline System)

Stage 1 establishes a reliable baseline:
- a single `scikit-learn` pipeline combines preprocessing and logistic regression
- training is config-driven and tracked in MLflow
- one serialized artifact (`churn_pipeline.pkl`) is produced for consistent inference

Run:

```bash
python -m training.train
```

Primary output:
- `models/artifacts/churn_pipeline.pkl`

## Stage 2 (Model Selection System)

Stage 2 upgrades the baseline into a structured model selection workflow:
- trains multiple candidate models under shared preprocessing
- evaluates and compares models with common metrics
- calibrates probabilities
- runs threshold evaluation
- selects and stores a deployment artifact

Candidate models:
- logistic regression
- random forest
- gradient boosting

Run:

```bash
python -m training.train_stage2
```

Primary outputs:
- per-model artifacts in `models/artifacts/`
- selected deployment artifact: `models/artifacts/deployed_pipeline.pkl`
- comparison/evaluation outputs in `reports/stage2/`

## Architecture Overview

BayesPilot follows a modular training-to-serving architecture:

1. Data generation and configuration
2. Training pipeline (`training/`)
3. Experiment tracking (MLflow)
4. Model artifacts (`models/artifacts/`)
5. Deployment API (`app/api/main.py`)
6. Decision logic (`app/services/decision.py`)
7. Monitoring and logs (`app/monitoring/`, `logs/predictions.jsonl`)

Inference flow:

```text
Request payload
-> deployed pipeline (preprocessing + model)
-> churn probability
-> decision tier
-> logged prediction and latency
-> API response
```

## Decision Layer (Why it matters)

A probability alone is not an operational decision.  
BayesPilot includes a decision layer that converts model output into action categories (for example, monitor vs review vs escalate).

This matters because it:
- makes model output actionable for retention teams
- creates a transparent policy surface that can be tested and revised
- supports alignment between ML performance and operational constraints

Current implementation uses fixed thresholds in `app/services/decision.py`.

## Monitoring and Evaluation

The system includes lightweight production-oriented monitoring:
- prediction logging to `logs/predictions.jsonl`
- latency tracking in the serving layer

Evaluation and reporting include:
- per-model metrics and comparison files in `reports/stage2/`
- selection record in `reports/stage2/selected_model.json`

## What makes this a senior-level project

BayesPilot goes beyond a notebook-style model demo by emphasizing:
- reproducible pipelines and artifact discipline
- controlled model comparison and deployment selection
- calibration and threshold analysis, not raw accuracy only
- explicit decision layer between prediction and action
- API serving with monitoring and test coverage
- clean modular boundaries across training, serving, and operations

## Current limitations and next steps

Current limitations:
- decision policy is threshold-based rather than explicit expected-value optimization
- monitoring is foundational and does not yet include drift or outcome feedback loops
- interpretability outputs are limited to current reporting artifacts

Practical next steps:
- introduce expected-value decision logic with business cost/benefit parameters
- add business-aware evaluation metrics for retention impact
- expand interpretability reporting for deployment decisions
- strengthen monitoring with risk signals, drift checks, and closed-loop outcomes

## Quick Start

Setup:

```bash
python -m venv venv_bayespilot
source venv_bayespilot/bin/activate
pip install -r requirements.txt
```

Generate synthetic dataset:

```bash
python scripts/generate_churn_data.py
```

Run API:

```bash
uvicorn app.api.main:app --reload
```

Run tests:

```bash
pytest
```
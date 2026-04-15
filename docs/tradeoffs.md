# BayesPilot Design Tradeoffs

## Why Tradeoffs Matter

BayesPilot is intentionally built as a decision-support system, not a benchmark-only modeling repo and not a full enterprise platform.  
The design choices are explicit in code paths such as `training/train_stage2.py`, `training/thresholds.py`, `app/services/decision.py`, `app/api/main.py`, and `scripts/check_prediction_drift.py`.

## Simpler Models vs Stronger Models

**Both sides**
- Simpler models like logistic regression are easier to inspect and usually faster at inference.
- Stronger non-linear models like random forest and gradient boosting can capture interactions that linear models miss.

**What BayesPilot chose**
- Stage 2 trains three candidates from `configs/stage2_model_config.yaml`: `logistic_regression`, `random_forest`, and `gradient_boosting`.
- All candidates run through the same preprocessing pipeline (`training/pipeline.py`) and optional calibration path in `training/train_stage2.py`.

**Why this choice fits**
- Using a shared pipeline removes preprocessing drift between models and makes comparisons in `reports/stage2/model_comparison.csv` defensible.
- Keeping logistic regression in the candidate set preserves a transparent baseline while still testing stronger alternatives.

**Downside accepted**
- Restricting candidate families to these three models may miss higher-performing architectures for complex feature interactions.

## Technical Metrics vs Business Metrics

**Both sides**
- AUC/F1/precision/recall reflect discrimination quality and class-balance behavior.
- Expected net benefit reflects whether intervention policy creates positive economic value.

**What BayesPilot chose**
- `training/thresholds.py` performs threshold sweeps (default 0.10 to 0.90, step 0.05) per model.
- For each threshold, it computes `expected_net_benefit = expected_loss_prevented - intervention_cost_total` using configured intervention cost, churn loss, and intervention success rate.
- `training/train_stage2.py` records each model's `best_threshold` and `max_expected_net_benefit`, then `training/compare.py` ranks models by `max_expected_net_benefit` first.
- AUC and F1 are used as tie-breakers in ranking and as supporting evidence in selection messaging.

**Why this choice fits**
- It aligns model selection with the action economics the system actually executes, not only classification quality.
- It enables a traceable artifact trail: `threshold_summary.csv`, `threshold_business_summary.csv`, and `selected_model.json` in `reports/stage2`.

**Downside accepted**
- Net-benefit ranking depends on business assumptions (cost, churn loss, success rate); if those are wrong, the selected model/threshold can be suboptimal even with strong AUC.

## Most Important Design Decision

BayesPilot prioritizes business value over raw model metrics.  
That single choice drives threshold sweeps in `training/thresholds.py`, candidate ranking in `training/compare.py`, and final deployment selection in `training/train_stage2.py`.  
Stage 2 reports are structured around `max_expected_net_benefit` and `best_threshold`, with AUC/F1 as tie-breakers rather than primary criteria.  
This keeps training outputs consistent with the downstream decision policy instead of optimizing for disconnected benchmark scores.

## Fixed Thresholds vs Expected Value Policies

**Both sides**
- Fixed thresholds (for example, "intervene if probability > 0.5") are easy to communicate and stable to operate.
- Expected value policies adapt actions to explicit economic assumptions and can justify intervention/no-intervention at any probability.

**What BayesPilot chose**
- Online decisions use `make_decision()` in `app/services/decision.py`, which computes `EV(action)` vs `EV(no_action)` and returns `intervene` only when `net_benefit > 0`.
- The API returns decision internals (`expected_value_action`, `expected_value_no_action`, `net_benefit`, `implied_probability_threshold`) so behavior is auditable.

**Why this choice fits**
- It avoids hard-coding a static threshold into API behavior and keeps policy logic explicit in one service module.
- It enables decision observability because EV quantities are also written to prediction logs.

**Downside accepted**
- Decision outputs are sensitive to assumed intervention success rate and cost parameters; policy quality degrades if assumptions drift from reality.

## Global Interpretability vs Local Explanations

**Both sides**
- Global importance summaries are cheap to compute and stable across requests.
- Local explanations (SHAP/LIME) explain a specific prediction but increase runtime and dependency complexity.

**What BayesPilot chose**
- Stage 2 writes model-level feature importance to `reports/stage2/feature_importance.csv`.
- The API loads that artifact at startup and returns top global features via `_build_explanation()` in `app/api/main.py`.

**Why this choice fits**
- This avoids per-request explanation computation and keeps `/predict` latency bounded by model inference plus lightweight formatting.
- It still provides transparent model-level context tied to a concrete report artifact.

**Downside accepted**
- Returned explanations are not case-level attributions; they cannot explain why one specific customer received a specific score.

## Lightweight Monitoring vs Production-Grade Observability

**Both sides**
- JSONL logging plus batch scripts is easy to run in any environment and has low operational overhead.
- Production observability stacks provide richer alerting, dashboards, and real-time anomaly handling.

**What BayesPilot chose**
- `app/monitoring/prediction_logger.py` appends each inference to `logs/predictions.jsonl` (including probability, action, EV values, explanation metadata, and latency).
- `scripts/check_prediction_drift.py` reads logs and runs heuristic checks on volume, probability range, intervention rate, and latency.

**Why this choice fits**
- Logging failures are intentionally swallowed in the API logger so inference remains available even when monitoring writes fail.
- The monitor script gives quick operational signals without requiring external infrastructure.

**Downside accepted**
- JSONL files and heuristic checks limit longitudinal analytics, automated alert routing, and formal drift testing compared with a dedicated observability platform.

## Simplicity and Modularity vs Infrastructure Complexity

**Both sides**
- Scripted modules and file-based artifacts are straightforward to debug and reproduce locally.
- Registry/orchestration/CI-CD platforms improve automation and governance at larger scale.

**What BayesPilot chose**
- Stage 2 training, evaluation, thresholding, selection, and artifact promotion are composed in `training/train_stage2.py`.
- Deployment uses a promoted artifact (`models/artifacts/deployed_pipeline.pkl`) plus report artifacts in `reports/stage2`, loaded directly by the FastAPI service.

**Why this choice fits**
- It avoids introducing orchestration and registry dependencies that are not required for the project scope.
- It enables deterministic, inspectable runs where training outputs and serving inputs are plain files.

**Downside accepted**
- Artifact lifecycle management is manual; there is no built-in version governance, approval workflow, or automated rollout/rollback mechanism.

## What BayesPilot Optimizes For

- Selection and thresholding aligned to expected economic value.
- Explicit decision policy via expected value equations, not hidden heuristics.
- Reproducible Stage 2 outputs (`model_comparison.csv`, `threshold_summary.csv`, `selected_model.json`).
- Modular boundaries between training, decisioning, API serving, and monitoring.
- Interview-defensible architecture where choices map directly to code and artifacts.

## What BayesPilot Intentionally Does Not Optimize For

- Distributed training/serving infrastructure.
- Streaming feature pipelines and event-time processing.
- Per-instance explainability tooling at inference time.
- Fully automated retraining, validation gates, and deployment orchestration.
- Enterprise observability (dashboards, paging, incident workflows, SLO enforcement).

## Summary

BayesPilot's core tradeoff is deliberate: optimize for business-value-aligned decisions with transparent, modular implementation.  
It accepts limits in scale automation and local explainability to keep model selection, decision logic, and monitoring tightly connected and auditable.

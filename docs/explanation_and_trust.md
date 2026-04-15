# Explanation and Trust in BayesPilot

## Why Explanation Matters

BayesPilot is designed as a decision-support system, not only a prediction system. In practice, probability outputs are more useful when users can understand the model's general behavior, decision assumptions, and limits. Explanation helps stakeholders:

- interpret model behavior at a high level,
- challenge decisions when business context suggests caution, and
- use predictions responsibly instead of treating them as automatic truth.

This is especially important for retention actions, where intervention cost, customer value, and business policy all influence the final decision.

## What BayesPilot Explains Today

BayesPilot currently provides two lightweight explanation levels.

### 1) Global Interpretability (Stage 2 Reports)

Stage 2 training creates global feature-importance outputs in `reports/stage2/feature_importance.csv` for supported models. This provides model-level visibility into which features are generally influential across many predictions.

This layer helps answer questions such as:

- "Which features matter most to this model overall?"
- "How do logistic regression and tree-based models differ in what they emphasize?"

### 2) Prediction-Time Explanation Context (API Response)

The `/predict` API response now includes an `explanation` object with global metadata for the deployed model. This gives users immediate context while reviewing prediction and decision outputs.

The explanation is intentionally lightweight and human-readable. It is designed to support interpretation, not to claim full per-case explainability.

## What the API Explanation Means

The API explanation should be interpreted as **global context** for the deployed model.

It means:

- the response identifies globally important features for the deployed model (when available),
- users get a quick summary of how the model generally behaves,
- the prediction is presented as part of a broader decision-support payload (probability, business decision logic, and explanation context).

Used correctly, this helps stakeholders understand broad drivers behind risk modeling without adding heavy explanation infrastructure.

## What the API Explanation Does Not Mean

The API explanation should **not** be interpreted as a case-specific attribution.

It does not mean:

- a specific feature "caused" a specific individual prediction,
- the output is equivalent to SHAP, LIME, or other local explanation methods,
- the system is providing a causal explanation of churn behavior,
- business judgment should be replaced by model output.

In short: BayesPilot provides model-level transparency, not per-record causal interpretation.

## Responsible Use Guidelines

Use BayesPilot as a support tool within operational and business review processes.

- Treat predictions as inputs to decisions, not final decisions by themselves.
- Combine model outputs with account context, customer history, and operational constraints.
- Revisit expected-value assumptions (cost, churn loss, intervention success rate) on a regular cadence.
- Monitor model quality and latency over time; investigate degradation before trusting unchanged policy thresholds.
- Avoid over-interpreting small differences in feature importance ranks.
- Keep a human-in-the-loop for high-impact actions, policy changes, or edge cases.

## Current Limitations

BayesPilot intentionally stays lightweight, which creates known limitations:

- Dataset realism: synthetic churn data may not represent production complexity or bias patterns.
- Economic assumptions: expected-value logic uses simplified assumptions that may drift from business reality.
- Explanation scope: global feature importance is not local attribution.
- Model drift risk: behavior may change after retraining, even with the same pipeline structure.
- Policy dependence: threshold choices and intervention strategy quality depend on assumptions that require periodic validation.

These limits are expected in a portfolio-focused system and should be acknowledged explicitly.

## What a Production System Would Add

A production deployment would typically extend this foundation with:

- local explanation support for case-level review (for example, SHAP-style attributions),
- automated drift detection dashboards for data, predictions, and outcomes,
- policy and threshold audit trails linked to business KPIs,
- explicit human review workflows for sensitive or high-value decisions,
- model/version metadata in every API response and decision log for traceability.

These additions improve governance, trust, and operational reliability at scale.

## Summary

BayesPilot now includes practical interpretability signals and prediction-time explanation context while staying lightweight. The system is transparent about what it can explain today and where its boundaries are. This balance is deliberate: provide useful trust-building context now, while clearly identifying what stronger production explainability and governance would require.

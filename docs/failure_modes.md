# Failure Modes in BayesPilot

## Why Failure Mode Thinking Matters

BayesPilot is a decision-support system, not an autonomous decision maker. In practice, failure can happen in data quality, model outputs, decision policy assumptions, serving performance, and monitoring coverage.  
The goal is early detection and responsible use, not perfection.

## Data Quality Issues

**What can go wrong**
- Missing or malformed inputs at prediction time.
- Schema mismatches between training-time and serving-time features.
- Unexpected category values (for example, unseen `region` values).
- Unrealistic numeric ranges (for example, extreme `usage` or `bill` values).

**Why it matters**
- Bad inputs can produce unstable probabilities or incorrect recommendations.
- Silent data issues can look like model drift when the root cause is input quality.

**How to detect it**
- Request validation and type checks in the API layer.
- Input sanity checks in logs (missing fields, invalid ranges, unknown categories).
- Periodic review of logged input distributions.

**Possible mitigation**
- Reject invalid requests with clear errors.
- Add guardrails for out-of-range values.
- Update preprocessing/encoding strategy for expected new categories.

## Feature Distribution Shift

**What can go wrong**
- Feature distributions for `usage`, `bill`, or `support_calls` move over time.
- Customer behavior changes due to seasonality, pricing, product changes, or market shifts.
- Production inputs drift away from training data characteristics.

**Why it matters**
- The model may still run but become less reliable in new regions of feature space.
- Decision rates can change unexpectedly even if code does not change.

**How to detect it**
- Compare recent logged feature summaries to training/reference ranges.
- Track rolling averages/quantiles by day or week.
- Use the lightweight drift script to watch prediction behavior shifts.

**Possible mitigation**
- Investigate upstream business or product changes.
- Refresh training data and retrain when shift is sustained.
- Re-evaluate thresholds and expected-value assumptions after retraining.

## Calibration Drift

**What can go wrong**
- Predicted probabilities stop matching true churn likelihood.
- The decision layer relies on probabilities that become miscalibrated.

**Why it matters**
- Even with stable ranking quality, poor calibration can reduce business value.
- Expected-value decisions become less trustworthy when probabilities are biased.

**How to detect it**
- Periodic calibration checks when outcome labels are available.
- Track realized churn by probability bands over time.
- Watch for policy instability (for example, too many interventions at moderate scores).

**Possible mitigation**
- Recalibrate probabilities or retrain the model.
- Temporarily adjust threshold/policy while calibration is being corrected.
- Add periodic calibration review to model governance cadence.

## Decision Policy Risk

**What can go wrong**
- Expected-value assumptions become outdated.
- Intervention cost increases.
- Churn loss estimates change.
- Intervention success rate drops.

**Why it matters**
- A technically accurate model can still drive poor business decisions.
- Net benefit can shrink or become negative without visible model failure.

**How to detect it**
- Monitor logged `expected_value_action`, `expected_value_no_action`, and `net_benefit`.
- Track intervention rate and compare with business capacity.
- Compare projected value vs observed campaign outcomes.

**Possible mitigation**
- Re-estimate policy parameters on a fixed cadence.
- Re-run threshold optimization with updated economics.
- Add business sign-off for major policy updates.

## Latency Degradation

**What can go wrong**
- Prediction latency increases.
- API responsiveness degrades under load.
- Resource contention or infrastructure bottlenecks emerge.

**Why it matters**
- Slow responses reduce operational usability and can cause timeouts.
- Degraded serving can hide or delay other monitoring signals.

**How to detect it**
- Track average/median/max `latency_ms` in prediction logs.
- Use warning thresholds in the drift monitoring script.
- Review latency by day and during peak traffic windows.

**Possible mitigation**
- Profile slow paths and optimize preprocessing/inference.
- Right-size deployment resources.
- Introduce basic request throttling or batching where appropriate.

## Logging and Monitoring Failures

**What can go wrong**
- Missing log lines from write failures.
- Malformed JSONL records.
- Drift checks not run regularly.
- Explanation artifacts missing or unavailable.

**Why it matters**
- Monitoring blind spots delay detection of real failures.
- Auditability and debugging become harder during incidents.

**How to detect it**
- Track log volume trends (unexpected drops/spikes).
- Count malformed lines in monitoring scripts.
- Check that scheduled monitoring scripts are executed.
- Validate presence of explanation artifacts and model metadata.

**Possible mitigation**
- Keep logging resilient and append-only, but monitor log health separately.
- Add lightweight runbooks/checklists for periodic monitoring tasks.
- Fail safely with explicit warnings when explanation artifacts are missing.

## Interpretability-Specific Risk

**What can go wrong**
- Users over-interpret global feature importance as case-level explanation.
- Users treat correlation-based importance as causal evidence.
- Explanation artifacts become stale after retraining.

**Why it matters**
- Misinterpretation can lead to overconfident or incorrect business actions.
- Trust can decrease if explanations conflict with current model behavior.

**How to detect it**
- Review usage guidance with stakeholders.
- Check that explanation artifacts match deployed model/version.
- Track whether explanation outputs are consistently available.

**Possible mitigation**
- Keep explicit documentation on explanation boundaries.
- Version explanation artifacts with model deployments.
- Add local explanation tooling in future production iterations.

## What BayesPilot Currently Does

BayesPilot already includes lightweight safeguards:
- Decision logic with explicit expected-value structure and validation-oriented design.
- Richer prediction logs capturing probability, decision outputs, rationale, latency, and model context.
- A lightweight drift-check script for summary statistics and heuristic warnings.
- Documentation on explanation scope, trust boundaries, and responsible use.
- Business-aware reporting and model selection based on expected net benefit.

## What a Production System Would Add

A production deployment would typically add:
- Automated alerting and dashboards.
- Stronger automated data validation and schema monitoring.
- Scheduled retraining triggers and model performance SLAs.
- Model registry/versioning with deployment approvals.
- Periodic calibration checks tied to real outcomes.
- Stronger audit trails, access controls, and incident response workflows.

## Summary

BayesPilot demonstrates practical failure-mode awareness while staying intentionally lightweight. The current approach is designed to catch obvious issues early, support responsible decision-making, and make system limits explicit. A full production system would build on this foundation with stronger automation, governance, and monitoring depth.

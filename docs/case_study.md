# BayesPilot — Case Study

## Problem

Customer churn prediction is often treated as a pure modeling task, but businesses do not act on probabilities alone; they act on interventions with costs and uncertain outcomes.  
The core problem in BayesPilot was to turn churn scoring into a decision-support workflow that can answer: **which customers should receive a retention action, and when is that action economically justified?**

Prediction alone is insufficient because a high-probability churn case is not automatically worth targeting if intervention cost exceeds expected value. BayesPilot reframes churn as a business decision problem, not just a classification problem.
The core idea is explicit: optimal action depends on expected economic value, not just predicted probability.

## Approach

I designed BayesPilot as a compact, production-style system with clear module boundaries:

- Train and evaluate multiple churn models under a shared pipeline.
- Compare candidates using both technical metrics and business-aware evaluation.
- Apply an expected value decision layer to convert probabilities into actions.
- Deploy the selected model behind a FastAPI service for inference.
- Log prediction and decision outputs, then run lightweight drift-style checks.

Prediction and decision logic are intentionally separated so the model and action policy can evolve independently.
The goal was to keep the system explainable, reproducible, and practical for discussion in interviews.

## System Design

BayesPilot follows a structured flow:

- **Training:** Build candidate models from churn data.
- **Selection:** Rank models using business-aware threshold analysis and supporting technical metrics.
- **Deployment:** Promote one artifact as the deployed model.
- **Inference:** Serve predictions through an API endpoint.
- **Decision:** Convert probability into `intervene` or `do_nothing` using expected value logic.
- **Logging and Monitoring:** Persist prediction/decision traces and run heuristic drift checks.

Two design elements are central:

- **Business-aware threshold optimization:** Evaluate threshold-dependent economic impact, not just classifier quality.
- **Explainability and monitoring hooks:** Return global feature-importance context in API responses and keep a record of behavior over time.

## Key Design Decisions

1. **Expected value policy over fixed threshold rules**  
   I chose explicit expected value decisioning so actions are justified by cost-loss assumptions instead of static probability cutoffs.

2. **Model selection by business value first**  
   Stage 2 prioritizes maximum expected net benefit and uses AUC/F1 as tie-breakers, aligning model choice with intervention economics.

3. **Global interpretability over per-instance explanations**  
   I used global feature importance to provide model-level transparency without introducing per-request explanation overhead.

4. **Lightweight monitoring over enterprise observability stack**  
   JSONL logging and drift heuristics were chosen to make post-deployment checks practical without adding infrastructure-heavy dependencies.

5. **Modular scripts over orchestration-heavy architecture**  
   The pipeline is split into clear training, selection, API, decision, and monitoring modules so behavior is easy to trace and defend.

## Results and Insights

- Introducing expected net benefit into Stage 2 can change model ranking compared with AUC/F1-only ordering.
- A high AUC model is not automatically the best deployment choice when intervention cost and expected recoverable loss are included.
- Threshold selection is an economic policy choice: changing threshold directly changes intervention volume, spend, and expected prevented churn loss.
- Separating prediction quality from action quality made tradeoffs auditable: the system can justify not only who is risky, but whether acting is worth it.

## Limitations

- Uses a synthetic-style dataset scope rather than a large real production dataset.
- Expected value policy currently assumes fixed intervention cost, constant churn loss, and a uniform intervention success rate.
- In real operations, those assumptions can vary by customer segment, channel, and time period, which can shift the optimal policy.
- Does not provide local per-customer explanations (for example, SHAP values at inference time).
- Monitoring is intentionally lightweight and heuristic-driven, not a full observability platform with automated alerting workflows.

## What I Would Do Next in Production

- Add local explanation support for case-level decision reviews.
- Build dashboard and alerting workflows for latency, drift, and intervention-rate anomalies.
- Add scheduled retraining/evaluation pipelines with explicit validation gates.
- Introduce model and policy versioning so each decision is traceable to artifact and assumption versions.
- Expand monitoring from heuristic checks to formal drift/statistical tests.

## Summary

BayesPilot is a decision-aware ML system where prediction and action policy are deliberately separated. Model evaluation and selection are grounded in expected economic impact rather than accuracy alone, so deployment choices reflect intervention value, not just classifier performance. This framing makes the system easier to defend in production-style discussions about policy, tradeoffs, and operational risk.

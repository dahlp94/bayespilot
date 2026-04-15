# Decision Layer Assumptions

This document describes the business assumptions used by the decision layer in BayesPilot's expected value (EV) framework.

## Core assumptions

### 1) `intervention_cost`

`intervention_cost` is the expected cost of taking a retention action for one customer (for example, discount, agent time, or outreach cost).

- Higher values make intervention less attractive.
- Lower values make intervention more likely.

### 2) `expected_loss_if_no_action`

`expected_loss_if_no_action` is the estimated business loss if a churn event occurs and no retention action is taken.

In code, this is represented as `churn_loss` with the same meaning.

- Higher values increase the value of preventing churn.
- Lower values reduce the economic case for intervention.

### 3) `retention_success_rate`

`retention_success_rate` is the probability that intervention successfully prevents churn for a customer who would otherwise churn.

- Higher values increase expected value of intervention.
- Lower values reduce intervention impact and may shift decisions toward no action.

## How assumptions influence decisions

The decision layer compares:

- `EV(no_action) = -probability_of_churn * expected_loss_if_no_action`
- `EV(action) = -intervention_cost - probability_of_churn * (1 - retention_success_rate) * expected_loss_if_no_action`

The system recommends:

- `intervene` when `EV(action) > EV(no_action)`
- `do_nothing` otherwise

This means decisions are not based on arbitrary thresholds alone; they are based on whether intervention creates positive expected economic value.

## Risks of incorrect assumptions

If assumptions are wrong, decisions can be systematically biased:

- Underestimated `intervention_cost` can cause over-intervention and budget waste.
- Overestimated `retention_success_rate` can create false confidence in campaign impact.
- Underestimated `expected_loss_if_no_action` can cause under-intervention and avoidable churn loss.
- Static assumptions may drift over time as customer behavior, pricing, and operations change.

## How these would be validated in production

Assumptions should be treated as measurable parameters, not fixed constants.

- Measure realized intervention cost from campaign and operations data.
- Estimate retention lift using controlled experiments (A/B tests) or robust causal evaluation.
- Estimate realized churn loss from downstream revenue and margin outcomes.
- Recalibrate assumptions on a recurring schedule and by customer segment.
- Monitor decision outcomes (intervene vs do_nothing) against realized business KPIs.

This closes the loop between model output, decision policy, and business outcomes.

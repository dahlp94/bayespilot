from __future__ import annotations


def make_decision(
    probability: float,
    intervention_cost: float = 25.0,
    churn_loss: float = 120.0,
    intervention_success_rate: float = 0.35,
) -> dict:
    """
    Make a retention decision using expected value.

    EV(no_action) = -probability * churn_loss
    EV(action) = -intervention_cost
                 - probability * (1 - intervention_success_rate) * churn_loss

    Decision rule:
    - intervene when net_benefit = EV(action) - EV(no_action) > 0
    - otherwise do_nothing
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1.")
    if not 0.0 <= intervention_success_rate <= 1.0:
        raise ValueError("intervention_success_rate must be between 0 and 1.")
    if intervention_cost < 0.0:
        raise ValueError("intervention_cost must be non-negative.")
    if churn_loss < 0.0:
        raise ValueError("churn_loss must be non-negative.")

    expected_value_no_action = -probability * churn_loss
    expected_value_action = (
        -intervention_cost
        - probability * (1.0 - intervention_success_rate) * churn_loss
    )

    net_benefit = expected_value_action - expected_value_no_action
    recommended_action = "intervene" if net_benefit > 0.0 else "do_nothing"

    denominator = intervention_success_rate * churn_loss
    implied_probability_threshold = (
        intervention_cost / denominator if denominator > 0.0 else None
    )

    ev_action_str = f"{expected_value_action:.4f}"
    ev_no_action_str = f"{expected_value_no_action:.4f}"
    if recommended_action == "intervene":
        rationale = (
            f"Intervene because EV(action)={ev_action_str} exceeds "
            f"EV(no_action)={ev_no_action_str}."
        )
    else:
        rationale = (
            f"Do nothing because EV(no_action)={ev_no_action_str} is greater than or "
            f"equal to EV(action)={ev_action_str}."
        )

    return {
        "recommended_action": recommended_action,
        "expected_value_action": round(expected_value_action, 4),
        "expected_value_no_action": round(expected_value_no_action, 4),
        "net_benefit": round(net_benefit, 4),
        "implied_probability_threshold": (
            round(implied_probability_threshold, 4)
            if implied_probability_threshold is not None
            else None
        ),
        "rationale": rationale,
    }


def make_ev_decision(
    probability: float,
    cost_of_intervention: float,
    cost_of_churn: float,
    success_rate_of_intervention: float,
) -> str:
    """
    Backward-compatible helper that returns only the action label.
    """
    decision = make_decision(
        probability=probability,
        intervention_cost=cost_of_intervention,
        churn_loss=cost_of_churn,
        intervention_success_rate=success_rate_of_intervention,
    )
    return decision["recommended_action"]

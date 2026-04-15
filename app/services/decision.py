DEFAULT_COST_OF_INTERVENTION = 40.0
DEFAULT_COST_OF_CHURN = 200.0
DEFAULT_SUCCESS_RATE_OF_INTERVENTION = 0.35


def make_ev_decision(
    p: float,
    cost_of_intervention: float,
    cost_of_churn: float,
    success_rate_of_intervention: float,
) -> tuple[str, float]:
    """
    Compute expected value and return a decision.

    EV(action) = p * success_rate_of_intervention * cost_of_churn
                 - cost_of_intervention
    EV(no_action) = 0

    Returns:
        tuple[str, float]: (decision, expected_value_of_action)
    """
    expected_value_action = (
        p * success_rate_of_intervention * cost_of_churn
        - cost_of_intervention
    )
    expected_value_no_action = 0.0
    decision = (
        "intervene"
        if expected_value_action > expected_value_no_action
        else "do_nothing"
    )
    return decision, expected_value_action


def make_decision(
    p: float,
    cost_of_intervention: float = DEFAULT_COST_OF_INTERVENTION,
    cost_of_churn: float = DEFAULT_COST_OF_CHURN,
    success_rate_of_intervention: float = DEFAULT_SUCCESS_RATE_OF_INTERVENTION,
) -> tuple[str, float]:
    """
    Public decision entrypoint used by the API.

    Uses default economic assumptions but allows overrides for
    scenario testing and future configuration wiring.
    """
    return make_ev_decision(
        p=p,
        cost_of_intervention=cost_of_intervention,
        cost_of_churn=cost_of_churn,
        success_rate_of_intervention=success_rate_of_intervention,
    )

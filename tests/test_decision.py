import pytest

from app.services.decision import make_decision, make_ev_decision


def test_make_decision_low_probability_recommends_do_nothing():
    decision = make_decision(probability=0.10)
    assert decision["recommended_action"] == "do_nothing"


def test_make_decision_high_probability_recommends_intervene():
    decision = make_decision(probability=0.90)
    assert decision["recommended_action"] == "intervene"


def test_make_decision_expected_value_calculations_are_correct():
    decision = make_decision(
        probability=0.8,
        intervention_cost=25.0,
        churn_loss=120.0,
        intervention_success_rate=0.35,
    )

    assert decision["expected_value_no_action"] == pytest.approx(-96.0)
    assert decision["expected_value_action"] == pytest.approx(-87.4)
    assert decision["net_benefit"] == pytest.approx(8.6)


def test_make_decision_near_threshold_changes_action_in_expected_direction():
    intervention_cost = 25.0
    churn_loss = 120.0
    intervention_success_rate = 0.35
    threshold = intervention_cost / (intervention_success_rate * churn_loss)
    epsilon = 1e-3
    below_probability = max(0.0, threshold - epsilon)
    above_probability = min(1.0, threshold + epsilon)

    just_below = make_decision(
        probability=below_probability,
        intervention_cost=intervention_cost,
        churn_loss=churn_loss,
        intervention_success_rate=intervention_success_rate,
    )
    just_above = make_decision(
        probability=above_probability,
        intervention_cost=intervention_cost,
        churn_loss=churn_loss,
        intervention_success_rate=intervention_success_rate,
    )

    assert just_below["recommended_action"] == "do_nothing"
    assert just_above["recommended_action"] == "intervene"


def test_make_decision_returns_expected_output_structure():
    decision = make_decision(probability=0.5)
    expected_keys = {
        "recommended_action",
        "expected_value_action",
        "expected_value_no_action",
        "net_benefit",
        "implied_probability_threshold",
        "rationale",
    }

    assert set(decision.keys()) == expected_keys


def test_make_ev_decision_returns_action_label_and_matches_make_decision():
    probability = 0.6
    intervention_cost = 25.0
    churn_loss = 120.0
    intervention_success_rate = 0.35

    full_decision = make_decision(
        probability=probability,
        intervention_cost=intervention_cost,
        churn_loss=churn_loss,
        intervention_success_rate=intervention_success_rate,
    )
    action = make_ev_decision(
        probability=probability,
        cost_of_intervention=intervention_cost,
        cost_of_churn=churn_loss,
        success_rate_of_intervention=intervention_success_rate,
    )

    assert isinstance(action, str)
    assert action == full_decision["recommended_action"]


def test_make_decision_raises_for_probability_below_zero():
    with pytest.raises(ValueError):
        make_decision(probability=-0.01)


def test_make_decision_raises_for_probability_above_one():
    with pytest.raises(ValueError):
        make_decision(probability=1.01)


def test_make_decision_raises_for_success_rate_below_zero():
    with pytest.raises(ValueError):
        make_decision(probability=0.5, intervention_success_rate=-0.1)


def test_make_decision_raises_for_success_rate_above_one():
    with pytest.raises(ValueError):
        make_decision(probability=0.5, intervention_success_rate=1.1)


def test_make_decision_raises_for_negative_intervention_cost():
    with pytest.raises(ValueError):
        make_decision(probability=0.5, intervention_cost=-1.0)


def test_make_decision_raises_for_negative_churn_loss():
    with pytest.raises(ValueError):
        make_decision(probability=0.5, churn_loss=-1.0)

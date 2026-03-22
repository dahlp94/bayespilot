from app.services.decision import make_decision


def test_low_risk():
    assert make_decision(0.1) == "low_risk_monitor"


def test_medium_risk():
    assert make_decision(0.5) == "medium_risk_review"


def test_high_risk():
    assert make_decision(0.9) == "high_risk_escalate"

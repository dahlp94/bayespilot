def make_decision(
    probability: float,
    low_threshold: float = 0.3,
    high_threshold: float = 0.7,
) -> str:
    if probability < low_threshold:
        return "low_risk_monitor"
    if probability < high_threshold:
        return "medium_risk_review"
    return "high_risk_escalate"

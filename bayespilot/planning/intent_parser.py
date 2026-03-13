from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ParsedIntent:
    intent: str
    explanation: str

def parse_intent(question: str) -> ParsedIntent:
    q = question.lower().strip()

    if any(word in q for word in ["influence", "affect", "associated", "relationship"]):
        return ParsedIntent(
            intent="coefficient_interpretation",
            explanation="User wants to understand which predictors are associated with the target.",
        )

    if any(word in q for word in ["predict", "probability", "likelihood"]):
        return ParsedIntent(
            intent="prediction",
            explanation="User wants predictive understanding of the target outcome.",
        )

    return ParsedIntent(
        intent="general_analysis",
        explanation="User is requesting a general Bayesian analysis of the selected target.",
    )
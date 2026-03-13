from dataclasses import dataclass
from typing import List, Literal

ModelType = Literal["bayesian_linear_regression", "bayesian_logistic_regression"]
TargetType = Literal["continuous", "binary"]

@dataclass
class ModelSpec:
    target: str
    predictors: List[str]
    model_type: ModelType
    target_type: TargetType
    question: str
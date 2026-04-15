from __future__ import annotations

import json
import os
from datetime import datetime, timezone


LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "predictions.jsonl")


def log_prediction(input_data: dict, output_data: dict) -> None:
    # Logging supports observability/debugging and should never fail requests.
    try:
        os.makedirs(LOG_DIR, exist_ok=True)

        output = output_data or {}
        decision = output.get("decision") or {}
        explanation = output.get("explanation") or {}

        expected_value_action = output.get("expected_value_action")
        if expected_value_action is None:
            expected_value_action = decision.get("expected_value_action")

        expected_value_no_action = output.get("expected_value_no_action")
        if expected_value_no_action is None:
            expected_value_no_action = decision.get("expected_value_no_action")

        net_benefit = output.get("net_benefit")
        if net_benefit is None:
            net_benefit = decision.get("net_benefit")

        implied_probability_threshold = output.get("implied_probability_threshold")
        if implied_probability_threshold is None:
            implied_probability_threshold = decision.get("implied_probability_threshold")

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": output.get("model_name"),
            "input": input_data if isinstance(input_data, dict) else None,
            "prediction_probability": output.get("prediction_probability")
            if output.get("prediction_probability") is not None
            else output.get("probability"),
            "recommended_action": output.get("recommended_action")
            if output.get("recommended_action") is not None
            else decision.get("recommended_action"),
            "rationale": output.get("rationale")
            if output.get("rationale") is not None
            else decision.get("rationale"),
            "expected_value_action": expected_value_action,
            "expected_value_no_action": expected_value_no_action,
            "net_benefit": net_benefit,
            "implied_probability_threshold": implied_probability_threshold,
            "explanation_type": explanation.get("explanation_type"),
            "explanation_summary": explanation.get("explanation_summary"),
            "latency_ms": output.get("latency_ms"),
        }

        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        # Intentionally swallow errors to keep inference API resilient.
        return
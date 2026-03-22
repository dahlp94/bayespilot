from __future__ import annotations

import json
import os
from datetime import datetime


LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "predictions.jsonl")


def log_prediction(input_data: dict, output_data: dict) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_data,
        "output": output_data,
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
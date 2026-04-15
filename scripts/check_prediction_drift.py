from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median


DEFAULT_LOG_PATH = "logs/predictions.jsonl"


def _to_float(value):
    try:
        numeric = float(value)
        if math.isnan(numeric):
            return None
        return numeric
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value):
    if not value:
        return None
    if isinstance(value, str) and value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _get_probability(record):
    probability = record.get("prediction_probability")
    if probability is None:
        probability = record.get("probability")
    return _to_float(probability)


def load_predictions(log_path: Path):
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    records = []
    malformed_lines = 0
    with log_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    records.append(payload)
                else:
                    malformed_lines += 1
            except json.JSONDecodeError:
                malformed_lines += 1

    if not records:
        raise ValueError(
            f"No valid prediction records found in {log_path}. "
            f"Malformed lines: {malformed_lines}."
        )
    return records, malformed_lines


def summarize_predictions(records):
    probabilities = [_get_probability(r) for r in records if _get_probability(r) is not None]
    latencies = [
        _to_float(r.get("latency_ms"))
        for r in records
        if _to_float(r.get("latency_ms")) is not None
    ]

    action_counter = Counter()
    for record in records:
        action = record.get("recommended_action")
        if action is None:
            decision = record.get("decision") or {}
            action = decision.get("recommended_action")
        if action is None:
            action = "unknown"
        action_counter[str(action)] += 1

    total_predictions = len(records)
    action_proportions = {
        action: count / total_predictions for action, count in action_counter.items()
    }

    prediction_summary = {
        "total_predictions": total_predictions,
        "average_prediction_probability": mean(probabilities) if probabilities else None,
        "median_prediction_probability": median(probabilities) if probabilities else None,
        "min_prediction_probability": min(probabilities) if probabilities else None,
        "max_prediction_probability": max(probabilities) if probabilities else None,
    }

    action_summary = {
        "count_by_recommended_action": dict(action_counter),
        "proportion_by_recommended_action": action_proportions,
    }

    latency_summary = {
        "average_latency_ms": mean(latencies) if latencies else None,
        "median_latency_ms": median(latencies) if latencies else None,
        "max_latency_ms": max(latencies) if latencies else None,
    }

    return {
        "prediction_summary": prediction_summary,
        "action_summary": action_summary,
        "latency_summary": latency_summary,
    }


def summarize_by_day(records):
    daily = defaultdict(lambda: {"count": 0, "probabilities": []})
    for record in records:
        timestamp = _parse_timestamp(record.get("timestamp"))
        if timestamp is None:
            continue
        day = timestamp.date().isoformat()
        daily[day]["count"] += 1
        probability = _get_probability(record)
        if probability is not None:
            daily[day]["probabilities"].append(probability)

    lines = []
    for day in sorted(daily.keys()):
        probs = daily[day]["probabilities"]
        avg_prob = mean(probs) if probs else None
        avg_prob_text = f"{avg_prob:.3f}" if avg_prob is not None else "N/A"
        lines.append(f"- {day}: count={daily[day]['count']}, avg_probability={avg_prob_text}")
    return lines


def detect_warnings(summary, min_volume, max_avg_latency_ms):
    warnings = []
    prediction_summary = summary["prediction_summary"]
    action_summary = summary["action_summary"]
    latency_summary = summary["latency_summary"]

    total_predictions = prediction_summary["total_predictions"]
    avg_probability = prediction_summary["average_prediction_probability"]
    avg_latency = latency_summary["average_latency_ms"]

    if total_predictions < min_volume:
        warnings.append(
            f"Low prediction volume: {total_predictions} records (threshold: {min_volume})."
        )
    if total_predictions > 100000:
        warnings.append(
            "Very high prediction volume detected (>100000). Verify traffic/source integrity."
        )

    if avg_probability is not None and (avg_probability < 0.20 or avg_probability > 0.80):
        warnings.append(
            f"Average prediction probability ({avg_probability:.3f}) is outside expected range [0.20, 0.80]."
        )

    intervene_rate = action_summary["proportion_by_recommended_action"].get("intervene")
    if intervene_rate is not None and (intervene_rate < 0.05 or intervene_rate > 0.80):
        warnings.append(
            f"Intervention rate ({intervene_rate:.3f}) is outside expected range [0.05, 0.80]."
        )

    if avg_latency is not None and avg_latency > max_avg_latency_ms:
        warnings.append(
            f"Average latency ({avg_latency:.2f} ms) exceeds threshold ({max_avg_latency_ms:.2f} ms)."
        )

    return warnings


def _fmt(value, digits=3):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Lightweight drift-like monitoring for BayesPilot prediction logs "
            "(simple heuristics, not formal drift testing)."
        )
    )
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH, help="Path to JSONL log file.")
    parser.add_argument(
        "--min-volume",
        type=int,
        default=20,
        help="Minimum expected prediction volume before warning.",
    )
    parser.add_argument(
        "--max-avg-latency-ms",
        type=float,
        default=50.0,
        help="Average latency warning threshold in milliseconds.",
    )
    args = parser.parse_args()

    try:
        records, malformed_lines = load_predictions(Path(args.log_path))
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return

    summary = summarize_predictions(records)
    warnings = detect_warnings(summary, args.min_volume, args.max_avg_latency_ms)
    daily_lines = summarize_by_day(records)

    prediction_summary = summary["prediction_summary"]
    action_summary = summary["action_summary"]
    latency_summary = summary["latency_summary"]

    print("=== BayesPilot Prediction Drift Check ===")
    print(f"Log file: {args.log_path}")
    print(f"Valid records: {prediction_summary['total_predictions']}")
    print(f"Malformed lines skipped: {malformed_lines}")
    print("")

    print("[Prediction Summary]")
    print(f"- total_predictions: {prediction_summary['total_predictions']}")
    print(
        f"- average_prediction_probability: {_fmt(prediction_summary['average_prediction_probability'])}"
    )
    print(
        f"- median_prediction_probability: {_fmt(prediction_summary['median_prediction_probability'])}"
    )
    print(f"- min_prediction_probability: {_fmt(prediction_summary['min_prediction_probability'])}")
    print(f"- max_prediction_probability: {_fmt(prediction_summary['max_prediction_probability'])}")
    print("")

    print("[Action Summary]")
    counts = action_summary["count_by_recommended_action"]
    proportions = action_summary["proportion_by_recommended_action"]
    if not counts:
        print("- No action records available.")
    else:
        for action in sorted(counts.keys()):
            print(
                f"- {action}: count={counts[action]}, proportion={_fmt(proportions.get(action))}"
            )
    print("")

    print("[Latency Summary]")
    print(f"- average_latency_ms: {_fmt(latency_summary['average_latency_ms'], digits=2)}")
    print(f"- median_latency_ms: {_fmt(latency_summary['median_latency_ms'], digits=2)}")
    print(f"- max_latency_ms: {_fmt(latency_summary['max_latency_ms'], digits=2)}")
    print("")

    if daily_lines:
        print("[By Day]")
        for line in daily_lines:
            print(line)
        print("")

    print("[Warnings]")
    if not warnings:
        print("- No warnings triggered.")
    else:
        for warning in warnings:
            print(f"- {warning}")

    print("")
    print(
        "Note: This is a lightweight heuristic monitor for obvious shifts; "
        "it is not a full drift detection or observability platform."
    )


if __name__ == "__main__":
    main()

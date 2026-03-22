from __future__ import annotations

import time


def now() -> float:
    return time.perf_counter()


def elapsed_ms(start_time: float) -> float:
    return round((time.perf_counter() - start_time) * 1000, 3)
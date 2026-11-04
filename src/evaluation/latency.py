import time
from dataclasses import dataclass
from typing import Dict

@dataclass
class LatencyMetrics:
    total_ms: float = 0.0
    vad_ms: float = 0.0
    liveness_ms: float = 0.0
    embedding_ms: float = 0.0
    verification_ms: float = 0.0

class LatencyTracker:
    """Utility to measure execution time of pipeline stages."""
    def __init__(self):
        self.metrics = LatencyMetrics()
        self._start_times: Dict[str, float] = {}

    def start(self, stage: str):
        self._start_times[stage] = time.perf_counter()

    def stop(self, stage: str):
        if stage in self._start_times:
            elapsed = (time.perf_counter() - self._start_times[stage]) * 1000.0
            setattr(self.metrics, f"{stage}_ms", elapsed)

    def record_total(self, start_time: float):
        self.metrics.total_ms = (time.perf_counter() - start_time) * 1000.0

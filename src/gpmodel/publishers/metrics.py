"""Metrics subscriber — accumulates perf samples and alert counts for summaries."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from gpmodel.core.events import AlertRaised, Event, PerfSampled


@dataclass
class MetricsSummary:
    frames: int = 0
    avg_fps: float = 0.0
    avg_latency_ms: float = 0.0
    perf_samples: int = 0
    alerts_total: int = 0
    alerts_by_rule: Counter[str] = field(default_factory=Counter)
    alerts_by_severity: Counter[str] = field(default_factory=Counter)


class MetricsSubscriber:
    """Aggregates runtime metrics for a post-run report."""

    def __init__(self) -> None:
        self._fps_sum = 0.0
        self._latency_sum = 0.0
        self._perf_samples = 0
        self._last_frame_count = 0
        self._alerts = 0
        self._alerts_by_rule: Counter[str] = Counter()
        self._alerts_by_severity: Counter[str] = Counter()

    def on_event(self, event: Event) -> None:
        if isinstance(event, PerfSampled) and event.sample is not None:
            self._fps_sum += event.sample.fps
            self._latency_sum += event.sample.latency_ms
            self._perf_samples += 1
            self._last_frame_count = event.sample.frame_count
        elif isinstance(event, AlertRaised):
            self._alerts += 1
            self._alerts_by_rule[event.rule_type] += 1
            self._alerts_by_severity[event.severity.value] += 1

    def summary(self) -> MetricsSummary:
        n = self._perf_samples
        return MetricsSummary(
            frames=self._last_frame_count,
            avg_fps=self._fps_sum / n if n else 0.0,
            avg_latency_ms=self._latency_sum / n if n else 0.0,
            perf_samples=n,
            alerts_total=self._alerts,
            alerts_by_rule=Counter(self._alerts_by_rule),
            alerts_by_severity=Counter(self._alerts_by_severity),
        )

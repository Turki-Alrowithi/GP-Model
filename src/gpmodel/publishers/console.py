"""Console subscriber — pretty-prints events to the terminal with `rich`."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.text import Text

from gpmodel.core.events import (
    AlertRaised,
    AlertSeverity,
    DetectionsReady,
    Event,
    PerfSampled,
    StreamStateChanged,
)

logger = logging.getLogger(__name__)

_SEVERITY_STYLE = {
    AlertSeverity.LOW: "green",
    AlertSeverity.MEDIUM: "yellow",
    AlertSeverity.HIGH: "orange1",
    AlertSeverity.CRITICAL: "bold red on black",
    AlertSeverity.OPERATIONAL: "cyan",
}


class ConsoleSubscriber:
    """Human-friendly console printer.

    By default shows alerts, perf snapshots, and stream state changes.
    Set `print_detections=True` to also print per-frame detection counts
    (verbose for real-time inference).
    """

    def __init__(
        self,
        print_detections: bool = False,
        console: Console | None = None,
    ) -> None:
        self._print_detections = print_detections
        self._console = console or Console()

    # ── Dispatch ───────────────────────────────────────────
    def on_event(self, event: Event) -> None:
        if isinstance(event, AlertRaised):
            self._print_alert(event)
        elif isinstance(event, PerfSampled):
            self._print_perf(event)
        elif isinstance(event, StreamStateChanged):
            self._print_state(event)
        elif isinstance(event, DetectionsReady) and self._print_detections:
            self._print_detections_summary(event)

    # ── Formatters ─────────────────────────────────────────
    def _print_alert(self, e: AlertRaised) -> None:
        style = _SEVERITY_STYLE.get(e.severity, "white")
        ts = e.timestamp.strftime("%H:%M:%S")
        tag = Text(f"[{e.severity.value}]", style=style)
        body = Text(f" {ts}  {e.stream_id}  {e.rule_type}: {e.title}")
        self._console.print(tag + body)
        if e.description:
            self._console.print(f"         {e.description}", style="dim")

    def _print_perf(self, e: PerfSampled) -> None:
        if e.sample is None:
            return
        s = e.sample
        self._console.print(
            f"[bold cyan]PERF[/bold cyan] {s.stream_id}  "
            f"{s.fps:5.1f} FPS  {s.latency_ms:6.1f} ms/frame  "
            f"(frames={s.frame_count}, dropped={s.dropped_frames})"
        )

    def _print_state(self, e: StreamStateChanged) -> None:
        color = {"opened": "green", "closed": "dim", "error": "red"}.get(e.state, "white")
        suffix = f" — {e.detail}" if e.detail else ""
        self._console.print(
            f"[{color}]● stream '{e.stream_id}' {e.state}[/{color}]{suffix}"
        )

    def _print_detections_summary(self, e: DetectionsReady) -> None:
        if not e.detections and not e.tracks:
            return
        frame_id = e.frame.frame_id if e.frame else "?"

        # When tracking is on we prefer a per-track summary — this is what
        # makes "it's the same person" visible to the operator.
        if e.tracks:
            by_class: dict[str, list[int]] = {}
            for t in e.tracks:
                by_class.setdefault(t.class_name, []).append(t.track_id)
            summary = ", ".join(
                f"{cls}#{','.join(str(i) for i in sorted(ids))}"
                for cls, ids in by_class.items()
            )
        else:
            counts: dict[str, int] = {}
            for d in e.detections:
                counts[d.class_name] = counts.get(d.class_name, 0) + 1
            summary = ", ".join(f"{k}={v}" for k, v in counts.items()) or "—"

        self._console.print(
            f"[dim]det[/dim] {e.stream_id} frame#{frame_id}  "
            f"detections={len(e.detections)} tracks={len(e.tracks)}  ({summary})"
        )

"""Alert dispatcher — the central Observer/Pub-Sub hub.

The pipeline publishes events here; any subscriber registered with
`subscribe()` gets a copy. Subscribers are isolated: an exception in
one never prevents delivery to the others.
"""

from __future__ import annotations

import logging
from threading import RLock

from gpmodel.core.events import Event
from gpmodel.core.interfaces import Subscriber

logger = logging.getLogger(__name__)


class AlertDispatcher:
    """Thread-safe, in-process event bus.

    Deliberately synchronous: detection pipelines emit at 25-30 Hz, and
    a well-behaved subscriber is cheap (append to a queue, write a log
    line). Heavy subscribers (network I/O) should buffer internally and
    hand off to their own worker thread.
    """

    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []
        self._lock = RLock()

    # ── Subscription ────────────────────────────────────────
    def subscribe(self, subscriber: Subscriber) -> None:
        with self._lock:
            if not any(s is subscriber for s in self._subscribers):
                self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: Subscriber) -> None:
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s is not subscriber]

    # ── Publishing ──────────────────────────────────────────
    def publish(self, event: Event) -> None:
        """Fan out an event to every subscriber. Failures are logged, not raised."""
        with self._lock:
            targets = list(self._subscribers)

        for sub in targets:
            try:
                sub.on_event(event)
            except Exception:
                logger.exception(
                    "Subscriber %s raised while handling %s", sub, type(event).__name__
                )

    # ── Introspection ──────────────────────────────────────
    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

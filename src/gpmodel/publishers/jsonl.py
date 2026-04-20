"""JSON-Lines file subscriber — one event per line, suitable for replay/analysis."""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from gpmodel.core.events import DetectionsReady, Event

logger = logging.getLogger(__name__)


class _EventEncoder(json.JSONEncoder):
    """JSON encoder that handles our domain types.

    - dataclasses → dicts of their fields
    - datetime → ISO-8601
    - StrEnum / Enum → value
    - numpy arrays → shape descriptor (pixel data never goes to disk)
    """

    def default(self, obj: object) -> object:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": {"shape": list(obj.shape), "dtype": str(obj.dtype)}}
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
        return super().default(obj)


class JSONLFileSubscriber:
    """Writes each received event as a single JSON line.

    File is opened in append mode so runs accumulate; the subscriber
    also tags each record with its event class name for easy grepping.
    """

    def __init__(
        self,
        path: str | Path,
        include_detection_frames: bool = False,
        flush_each: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._include_detection_frames = include_detection_frames
        self._flush_each = flush_each
        self._fp: TextIO = self.path.open("a", encoding="utf-8")

    # ── Dispatch ───────────────────────────────────────────
    def on_event(self, event: Event) -> None:
        # High-volume DetectionsReady is gated behind an explicit flag so
        # real-time inference doesn't saturate disk.
        if isinstance(event, DetectionsReady) and not self._include_detection_frames:
            return

        record = self._to_record(event)
        self._fp.write(json.dumps(record, cls=_EventEncoder) + "\n")
        if self._flush_each:
            self._fp.flush()

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()

    def __enter__(self) -> JSONLFileSubscriber:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Internals ──────────────────────────────────────────
    @staticmethod
    def _to_record(event: Event) -> dict[str, Any]:
        return {"type": type(event).__name__, "event": event}

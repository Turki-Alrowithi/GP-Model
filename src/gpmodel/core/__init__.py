"""Core domain: types, events, interfaces, dispatcher."""

from gpmodel.core.dispatcher import AlertDispatcher
from gpmodel.core.events import (
    AlertRaised,
    AlertSeverity,
    DetectionsReady,
    Event,
    PerfSampled,
    StreamStateChanged,
)
from gpmodel.core.interfaces import Detector, Subscriber, Tracker, VideoSource
from gpmodel.core.types import BBox, Detection, Frame, PerfSample, Track

__all__ = [
    "AlertDispatcher",
    "AlertRaised",
    "AlertSeverity",
    "BBox",
    "Detection",
    "DetectionsReady",
    "Detector",
    "Event",
    "Frame",
    "PerfSample",
    "PerfSampled",
    "StreamStateChanged",
    "Subscriber",
    "Track",
    "Tracker",
    "VideoSource",
]

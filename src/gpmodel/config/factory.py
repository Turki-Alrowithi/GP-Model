"""Factory functions — build runtime components from validated config.

Everything that instantiates a concrete class from config lives here,
so the rest of the code can stay decoupled from the config schema.
"""

from __future__ import annotations

from gpmodel.config.schema import (
    AppConfig,
    ByteTrackConfig,
    CrowdRuleConfig,
    FileConfig,
    GeofenceRuleConfig,
    GeofenceZoneConfig,
    PublishersConfig,
    RtspConfig,
    RulesConfig,
    SourceConfig,
    WebcamConfig,
    YoloConfig,
)
from gpmodel.core.dispatcher import AlertDispatcher
from gpmodel.core.events import AlertSeverity
from gpmodel.core.interfaces import Detector, Subscriber, Tracker, VideoSource
from gpmodel.detectors.yolo import YoloDetector
from gpmodel.pipeline.engine import InferenceEngine
from gpmodel.publishers.console import ConsoleSubscriber
from gpmodel.publishers.jsonl import JSONLFileSubscriber
from gpmodel.publishers.metrics import MetricsSubscriber
from gpmodel.rules.base import RulesEngine
from gpmodel.rules.crowd import CrowdRule
from gpmodel.rules.geofence import Geofence, GeofenceRule
from gpmodel.sources.file import FileSource
from gpmodel.sources.rtsp import RtspSource
from gpmodel.sources.webcam import WebcamSource
from gpmodel.trackers.bytetrack import ByteTrackTracker


def build_source(cfg: SourceConfig, stream_id: str) -> VideoSource:
    match cfg:
        case WebcamConfig():
            return WebcamSource(
                stream_id=stream_id,
                device_index=cfg.device_index,
                width=cfg.width,
                height=cfg.height,
                fps=cfg.fps,
            )
        case FileConfig():
            return FileSource(path=cfg.path, stream_id=stream_id, loop=cfg.loop)
        case RtspConfig():
            return RtspSource(
                url=cfg.url,
                stream_id=stream_id,
                transport=cfg.transport,
                reconnect_delay_s=cfg.reconnect_delay_s,
                max_reconnect_delay_s=cfg.max_reconnect_delay_s,
            )


def build_detector(cfg: YoloConfig) -> Detector:
    return YoloDetector(
        weights=cfg.weights,
        device=cfg.device,
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        iou=cfg.iou,
        classes=cfg.classes,
        half=cfg.half,
    )


def build_tracker(cfg: ByteTrackConfig) -> Tracker | None:
    if not cfg.enabled:
        return None
    return ByteTrackTracker(
        fps=cfg.fps,
        track_activation_threshold=cfg.track_activation_threshold,
        lost_track_buffer=cfg.lost_track_buffer,
        minimum_matching_threshold=cfg.minimum_matching_threshold,
        minimum_consecutive_frames=cfg.minimum_consecutive_frames,
    )


def _zone_to_geofence(cfg: GeofenceZoneConfig) -> Geofence:
    return Geofence(
        name=cfg.name,
        points=tuple((p[0], p[1]) for p in cfg.points),
        normalized=cfg.normalized,
    )


def build_geofence_rule(cfg: GeofenceRuleConfig) -> GeofenceRule | None:
    if not cfg.enabled or not cfg.zones:
        return None
    return GeofenceRule(
        zones=[_zone_to_geofence(z) for z in cfg.zones],
        classes=frozenset(cfg.classes),
        severity=AlertSeverity(cfg.severity),
        cooldown_s=cfg.cooldown_s,
        foot_point=cfg.foot_point,
    )


def build_crowd_rule(cfg: CrowdRuleConfig) -> CrowdRule | None:
    if not cfg.enabled:
        return None
    return CrowdRule(
        threshold=cfg.threshold,
        zone=_zone_to_geofence(cfg.zone) if cfg.zone is not None else None,
        classes=frozenset(cfg.classes),
        min_duration_s=cfg.min_duration_s,
        severity=AlertSeverity(cfg.severity),
        cooldown_s=cfg.cooldown_s,
    )


def build_rules(cfg: RulesConfig) -> RulesEngine | None:
    engine = RulesEngine()
    for rule in (build_geofence_rule(cfg.geofence), build_crowd_rule(cfg.crowd)):
        if rule is not None:
            engine.add(rule)
    return engine if engine.rules() else None


def build_publishers(
    cfg: PublishersConfig,
) -> tuple[list[Subscriber], MetricsSubscriber | None]:
    """Return the full subscriber list plus the metrics one (for summary access)."""
    subs: list[Subscriber] = []
    metrics: MetricsSubscriber | None = None

    if cfg.console.enabled:
        subs.append(ConsoleSubscriber(print_detections=cfg.console.print_detections))
    if cfg.jsonl.enabled:
        subs.append(
            JSONLFileSubscriber(
                path=cfg.jsonl.path,
                include_detection_frames=cfg.jsonl.include_detection_frames,
            )
        )
    if cfg.metrics.enabled:
        metrics = MetricsSubscriber()
        subs.append(metrics)

    return subs, metrics


def build_engine(
    cfg: AppConfig, dispatcher: AlertDispatcher
) -> tuple[InferenceEngine, MetricsSubscriber | None]:
    """Build everything from a validated AppConfig, register subscribers."""
    source = build_source(cfg.stream.source, cfg.stream.id)
    detector = build_detector(cfg.detector)
    tracker = build_tracker(cfg.tracker)
    rules = build_rules(cfg.rules)

    subscribers, metrics = build_publishers(cfg.publishers)
    for sub in subscribers:
        dispatcher.subscribe(sub)

    engine = InferenceEngine(
        stream_id=cfg.stream.id,
        source=source,
        detector=detector,
        dispatcher=dispatcher,
        tracker=tracker,
        rules=rules,
        perf_window=cfg.perf.window,
        perf_emit_every=cfg.perf.emit_every,
    )
    return engine, metrics

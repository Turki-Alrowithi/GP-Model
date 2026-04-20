"""Face encoder — wraps InsightFace's FaceAnalysis for identity lookup.

Pluggable via a FaceEncoder Protocol so tests can swap in a fake
encoder and the reid pipeline stays isolated from the heavyweight
InsightFace model at unit-test time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FaceEmbedding:
    """A detected face — bbox in absolute pixels + its L2-normalized embedding."""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    embedding: np.ndarray  # shape (D,), dtype float32, unit-norm
    det_score: float


@runtime_checkable
class FaceEncoder(Protocol):
    """Detect and embed faces in a BGR image."""

    def encode(self, image: np.ndarray) -> list[FaceEmbedding]: ...


class InsightFaceEncoder:
    """InsightFace (`buffalo_l`) via onnxruntime.

    Lazy-loads the FaceAnalysis pipeline on first use so tests that
    don't actually need face inference aren't taxed by the model load.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        providers: tuple[str, ...] | None = None,
    ) -> None:
        self.model_name = model_name
        self.det_size = det_size
        # CoreML EP first on Apple Silicon for the detection + recognition ONNX
        # graphs; CPU fallback so we stay runnable everywhere.
        self.providers = providers or ("CoreMLExecutionProvider", "CPUExecutionProvider")
        self._app: object | None = None

    # ── Lifecycle ──────────────────────────────────────────
    def _get_app(self) -> object:
        if self._app is not None:
            return self._app
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError("insightface is not installed. Run `uv sync --extra face`.") from e

        app = FaceAnalysis(name=self.model_name, providers=list(self.providers))
        app.prepare(ctx_id=0, det_size=self.det_size)
        logger.info("InsightFace ready (model=%s, providers=%s)", self.model_name, self.providers)
        self._app = app
        return app

    # ── FaceEncoder API ────────────────────────────────────
    def encode(self, image: np.ndarray) -> list[FaceEmbedding]:
        app = self._get_app()
        faces = app.get(image)  # type: ignore[attr-defined]
        out: list[FaceEmbedding] = []
        for f in faces:
            bbox = tuple(float(v) for v in f.bbox.tolist())
            emb = f.normed_embedding.astype(np.float32)
            out.append(
                FaceEmbedding(
                    bbox=bbox,  # type: ignore[arg-type]
                    embedding=emb,
                    det_score=float(getattr(f, "det_score", 0.0)),
                )
            )
        return out

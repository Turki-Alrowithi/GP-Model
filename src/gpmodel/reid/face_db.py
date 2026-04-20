"""Staff face database — enrol authorized people from a folder tree.

Folder layout:

    data/authorized/
    ├── alice/
    │   ├── 01.jpg
    │   └── 02.jpg
    └── bob/
        └── portrait.png

On load, each image is encoded by a FaceEncoder; each person ends up
with one or more 512-d embeddings. At query time, the closest staff
cosine-similarity score is returned — above `threshold` it's a match.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from gpmodel.reid.encoder import FaceEncoder

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True, slots=True)
class StaffMatch:
    """Result of looking up a probe embedding against the staff DB."""

    name: str
    similarity: float  # cosine


@dataclass
class StaffFaceDB:
    """In-memory DB of `name -> list[embedding]`, with a match threshold."""

    encoder: FaceEncoder
    threshold: float = 0.45  # cosine threshold for "same person"
    _embeddings: dict[str, list[np.ndarray]] = field(default_factory=dict, init=False)

    # ── Enrollment ─────────────────────────────────────────
    def enroll_directory(self, root: str | Path) -> int:
        """Walk `root/<name>/*.jpg` and enroll every face found.

        Returns the number of embeddings added.
        """
        root_path = Path(root)
        if not root_path.exists():
            logger.warning("Authorized directory does not exist: %s", root_path)
            return 0

        added = 0
        for person_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
            name = person_dir.name
            for img_path in sorted(person_dir.iterdir()):
                if img_path.suffix.lower() not in _IMAGE_SUFFIXES:
                    continue
                added += self._enroll_one(name, img_path)
        logger.info("Enrolled %d embeddings across %d names", added, len(self._embeddings))
        return added

    def _enroll_one(self, name: str, img_path: Path) -> int:
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Could not read %s — skipping", img_path)
            return 0
        faces = self.encoder.encode(image)
        if not faces:
            logger.warning("No face found in %s — skipping", img_path)
            return 0
        # Use the highest-confidence face per photo (enrollment photos should
        # be single-subject; if multiple faces exist we trust the detector).
        best = max(faces, key=lambda f: f.det_score)
        self._embeddings.setdefault(name, []).append(best.embedding)
        return 1

    # ── Query ──────────────────────────────────────────────
    def match(self, embedding: np.ndarray) -> StaffMatch | None:
        """Return the best-scoring staff member above `threshold`, else None."""
        if not self._embeddings:
            return None
        probe = embedding / (np.linalg.norm(embedding) + 1e-9)

        best_name: str | None = None
        best_sim = -1.0
        for name, embs in self._embeddings.items():
            # embs are already normalized by InsightFace; no need to renormalize.
            sims = np.array([float(np.dot(probe, e)) for e in embs])
            top = float(sims.max()) if sims.size else -1.0
            if top > best_sim:
                best_sim = top
                best_name = name

        if best_name is None or best_sim < self.threshold:
            return None
        return StaffMatch(name=best_name, similarity=best_sim)

    # ── Introspection ──────────────────────────────────────
    @property
    def size(self) -> int:
        return sum(len(v) for v in self._embeddings.values())

    @property
    def names(self) -> list[str]:
        return sorted(self._embeddings.keys())

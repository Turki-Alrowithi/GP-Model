"""Shared fixtures for rule tests."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from gpmodel.core.types import Frame


@pytest.fixture
def sample_frame() -> Frame:
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    return Frame(
        stream_id="cam-1",
        frame_id=1,
        timestamp=datetime.now(UTC),
        image=img,
    )

"""Tests for FileSource — uses the bundled sample clip."""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import pytest

from gpmodel.core.interfaces import VideoSource
from gpmodel.sources.file import FileSource

SAMPLE = Path(__file__).resolve().parents[2] / "assets" / "samples" / "drone_01.mp4"


@pytest.fixture
def sample_path() -> Path:
    if not SAMPLE.exists():
        pytest.skip(f"sample clip missing: {SAMPLE}")
    return SAMPLE


def test_is_video_source(sample_path: Path) -> None:
    src = FileSource(sample_path)
    assert isinstance(src, VideoSource)


def test_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FileSource(tmp_path / "nope.mp4")


def test_yields_frames_with_monotonic_ids(sample_path: Path) -> None:
    with FileSource(sample_path) as src:
        frames = list(islice(src.frames(), 5))

    assert len(frames) == 5
    assert [f.frame_id for f in frames] == [1, 2, 3, 4, 5]
    for f in frames:
        assert f.stream_id.startswith("file:")
        assert f.image is not None
        assert f.image.ndim == 3
        assert f.image.shape[2] == 3  # BGR


def test_loops_when_requested(sample_path: Path) -> None:
    # Exhaust once, verify we keep getting frames on loop
    with FileSource(sample_path, loop=True) as src:
        frames = list(islice(src.frames(), 500))
    # The sample clip is short (~192 frames); looping lets us exceed that
    assert len(frames) == 500


def test_context_manager_closes_capture(sample_path: Path) -> None:
    src = FileSource(sample_path)
    with src:
        assert src.is_open
    assert not src.is_open


def test_close_is_idempotent(sample_path: Path) -> None:
    src = FileSource(sample_path)
    src.open()
    src.close()
    src.close()  # must not raise
    assert not src.is_open

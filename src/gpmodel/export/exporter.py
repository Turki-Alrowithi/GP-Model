"""Model export — PyTorch weights → CoreML / ONNX / TorchScript.

Thin wrapper around Ultralytics' built-in `.export()` so we can call
it uniformly and route output into the project's `models/` directory.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

ExportFormat = Literal["coreml", "onnx", "torchscript"]

_FORMAT_SUFFIX: dict[str, str] = {
    "coreml": ".mlpackage",
    "onnx": ".onnx",
    "torchscript": ".torchscript",
}

# coremltools' PyTorch frontend lags ~1-2 minor versions behind torch.
# `coremltools<=8.3` fails on torch>=2.6 with
#     TypeError: only 0-dimensional arrays can be converted to Python scalars
# See https://github.com/apple/coremltools/issues/2521. Until coremltools
# 9.x catches up, the CoreML export requires an older torch in a separate
# venv. We surface the mismatch with a clear error instead of a mid-export
# traceback from the C++ converter.
_MAX_TORCH_FOR_COREML = (2, 6)


class CoreMLVersionError(RuntimeError):
    """Raised when the current torch version is known-incompatible with coremltools."""


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Where the exported model ended up plus the format it was written in."""

    path: Path
    format: str
    weights: Path
    imgsz: int
    half: bool


def export_model(
    weights: str | Path,
    fmt: ExportFormat,
    output_dir: str | Path = Path("models"),
    imgsz: int = 640,
    half: bool = False,
    nms: bool = True,
) -> ExportResult:
    """Export YOLO weights to `fmt` and move the result into `output_dir`.

    Ultralytics writes the export artefact next to the input `.pt` by
    default — we move it into `models/` afterwards so the repo layout
    stays predictable.

    Raises:
        FileNotFoundError: if the source weights don't exist.
        ImportError: if the extra needed for the chosen format isn't installed.
        RuntimeError: if Ultralytics refused the export.
    """
    from ultralytics import YOLO  # type: ignore[attr-defined]  # heavy import

    src = Path(weights)
    if not src.exists():
        raise FileNotFoundError(f"Weights not found: {src}")
    if fmt == "coreml":
        _check_coreml_compat()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s for export to %s (imgsz=%d half=%s)", src, fmt, imgsz, half)
    model = YOLO(str(src))

    exported_raw = model.export(format=fmt, imgsz=imgsz, half=half, nms=nms)
    # Ultralytics returns a str/Path pointing at the artefact it produced.
    produced = Path(str(exported_raw))
    if not produced.exists():
        raise RuntimeError(f"Export reported success but output is missing: {produced}")

    target = out_dir / (src.stem + _FORMAT_SUFFIX[fmt])
    _safe_move(produced, target)

    logger.info("Exported %s → %s", src.name, target)
    return ExportResult(path=target, format=fmt, weights=src, imgsz=imgsz, half=half)


def _safe_move(src: Path, dest: Path) -> None:
    """Move src onto dest, overwriting any pre-existing export at dest."""
    if dest.exists():
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    shutil.move(str(src), str(dest))


def _check_coreml_compat() -> None:
    """Refuse the export if torch is newer than what coremltools can parse."""
    import torch

    version_tuple = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
    if version_tuple >= _MAX_TORCH_FOR_COREML:
        raise CoreMLVersionError(
            "CoreML export is currently blocked by an upstream coremltools "
            f"regression on torch>={_MAX_TORCH_FOR_COREML[0]}.{_MAX_TORCH_FOR_COREML[1]} "
            f"(you are on torch {torch.__version__}).\n\n"
            "Workaround — create a separate venv for the export:\n"
            "    uv venv --python 3.11 .venv-coreml\n"
            "    source .venv-coreml/bin/activate\n"
            "    uv pip install 'torch<2.6' 'torchvision<0.21' "
            "'ultralytics>=8.3' 'coremltools==8.3.*'\n"
            "    python apps/export.py export --weights <your.pt> --format coreml\n\n"
            "Track upstream progress: https://github.com/apple/coremltools/issues/2521"
        )

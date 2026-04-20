#!/usr/bin/env python3
"""
download_model.py — Download YOLOv8n ONNX model for laptop testing.

Downloads from Ultralytics GitHub releases. No PyTorch needed.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

# YOLOv8n ONNX — pre-exported, ~12MB
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "yolov8n.onnx"


def download_with_progress(url: str, dest: Path) -> None:
    """Download a file with progress bar."""
    print(f"Downloading: {url}")
    print(f"        To: {dest}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            bar_len = 40
            filled = int(bar_len * percent / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  [{bar}] {percent:5.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)", end="", flush=True)
        else:
            mb_done = downloaded / (1024 * 1024)
            print(f"\r  Downloaded: {mb_done:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=progress_hook)
    print()  # newline after progress bar


def verify_onnx(path: Path) -> bool:
    """Quick check that the file is a valid ONNX model."""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        print(f"  ✓ Model verified: input={inp.name}, shape={inp.shape}, dtype={inp.type}")
        return True
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        return False


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"Model already exists: {MODEL_PATH} ({size_mb:.1f} MB)")
        if verify_onnx(MODEL_PATH):
            print("Ready to go! ✓")
            return
        else:
            print("Model appears corrupted, re-downloading...")
            MODEL_PATH.unlink()

    print("=" * 60)
    print("  Downloading YOLOv8n ONNX model for laptop testing")
    print("=" * 60)
    print()

    try:
        download_with_progress(MODEL_URL, MODEL_PATH)
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nAlternative: manually download from:")
        print(f"  {MODEL_URL}")
        print(f"  Save to: {MODEL_PATH}")
        sys.exit(1)

    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"\nDownloaded: {size_mb:.1f} MB")

    if verify_onnx(MODEL_PATH):
        print("\n✓ Model ready! Run the demo with:")
        print("  python tools/demo_laptop.py")
    else:
        print("\n✗ Model verification failed. Try downloading manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()

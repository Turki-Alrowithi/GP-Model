#!/usr/bin/env python3
"""Thin shim over `gpmodel.cli:main` for direct script execution.

Prefer `uv run gpmodel --config configs/laptop.yaml` once the project
is installed; this script exists so `python apps/run_inference.py`
still works during development without activating the venv.
"""

from __future__ import annotations

import sys

from gpmodel.cli import main

if __name__ == "__main__":
    sys.exit(main())

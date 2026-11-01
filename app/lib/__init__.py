"""Shared helpers for the ECG denoising Streamlit app.

Importing this package puts the repository root and the ``app`` directory on
``sys.path`` so that pages can use ``from lib...`` and ``from src...`` no matter
which file Streamlit happens to be executing.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR: Path = Path(__file__).resolve().parents[1]
"""Directory holding ``Home.py`` and ``pages/``."""

REPO_ROOT: Path = APP_DIR.parent
"""Repository root; contains ``src/`` and ``data/``."""

for _entry in (str(REPO_ROOT), str(APP_DIR)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

__all__ = ["APP_DIR", "REPO_ROOT"]

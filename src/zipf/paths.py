"""Filesystem locations. Every path in the project is resolved here, never hardcoded."""

from __future__ import annotations

import os
from pathlib import Path

# The project root is three parents up from this file: src/zipf/paths.py -> repo root.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_ROOT: Path = Path(os.environ.get("ZIPF_DATA_ROOT", REPO_ROOT / "data"))
INPUT_DIR: Path = DATA_ROOT / "input"
INTERIM_DIR: Path = DATA_ROOT / "interim"
OUTPUT_DIR: Path = DATA_ROOT / "output"
ASSETS_DIR: Path = REPO_ROOT / "assets"

#: Claude Code's transcript store. Read-only input: never written to, never rewritten.
CLAUDE_PROJECTS_DIR: Path = Path(
    os.environ.get("ZIPF_CLAUDE_PROJECTS", Path.home() / ".claude" / "projects")
)


def ensure_dirs() -> None:
    """Create the data tri-split if it is missing. Safe to call repeatedly."""
    for directory in (INPUT_DIR, INTERIM_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

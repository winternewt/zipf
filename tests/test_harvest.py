"""Harvesting the transcript store.

The depth tests are the regression for a real defect: transcripts sit at three different
nesting depths, and deriving the project by counting parents up from the file returns a
session uuid for the nested cases. Those uuids then look like distinct projects, which
silently multiplied the project count.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zipf.harvest import iter_transcripts, project_of


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def test_project_is_the_first_component_at_every_observed_depth(tmp_path: Path) -> None:
    for relative in (
        "myproject/session.jsonl",
        "myproject/865aa2a0-9274-42ac-aabd-f26a8c0feafa/session.jsonl",
        "myproject/865aa2a0-9274-42ac-aabd-f26a8c0feafa/subagents/agent-a879e86.jsonl",
    ):
        assert project_of(tmp_path / relative, tmp_path) == "myproject"


def test_iter_transcripts_skips_the_memory_directory(tmp_path: Path) -> None:
    _write(tmp_path / "proj" / "real.jsonl", [{"type": "x"}])
    _write(tmp_path / "proj" / "memory" / "note.jsonl", [{"type": "x"}])
    found = [p.name for p in iter_transcripts(tmp_path)]
    assert found == ["real.jsonl"]


def test_iter_transcripts_is_deterministically_ordered(tmp_path: Path) -> None:
    for name in ("c.jsonl", "a.jsonl", "b.jsonl"):
        _write(tmp_path / "proj" / name, [{"type": "x"}])
    first = [p.name for p in iter_transcripts(tmp_path)]
    assert first == sorted(first)
    assert first == [p.name for p in iter_transcripts(tmp_path)]


def test_missing_store_is_an_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="transcript store"):
        list(iter_transcripts(tmp_path / "nope"))

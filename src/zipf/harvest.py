"""Harvest assistant prose from the local Claude Code transcript store.

The store is read-only input. This module opens files there and never writes to, moves or
rewrites one: it is the user's session history and the only copy.

Two strata come out, kept separate rather than pooled:

``claude_main``
    Text the assistant addressed to the user. This is the target corpus.
``claude_sidechain``
    Text a subagent returned to its parent agent. Same models, different audience, so it acts
    as a control that distinguishes an addressed-to-a-human tic from an intrinsic one.

Only ``text`` content blocks carry prose. ``thinking`` blocks in this store are
signature-only — the plaintext is not retained — so reasoning text cannot be analysed, and a
run that reports on it would be reporting on nothing.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from zipf.paths import CLAUDE_PROJECTS_DIR, INTERIM_DIR

logger = logging.getLogger(__name__)

#: Subdirectories of a project dir that hold something other than transcripts.
_NON_TRANSCRIPT_DIRS = frozenset({"memory"})

DOCUMENTS_PARQUET = INTERIM_DIR / "claude_documents.parquet"


def iter_transcripts(root: Path = CLAUDE_PROJECTS_DIR) -> Iterator[Path]:
    """Yield every transcript file, deterministically ordered.

    Sidechain sessions nest one level deeper in a per-session subdirectory; ``memory/`` holds
    memory files rather than transcripts and is skipped.
    """
    if not root.exists():
        raise FileNotFoundError(
            f"Claude Code transcript store not found at {root}. "
            "Set ZIPF_CLAUDE_PROJECTS if it lives elsewhere."
        )
    for path in sorted(root.rglob("*.jsonl")):
        if _NON_TRANSCRIPT_DIRS & set(path.relative_to(root).parts):
            continue
        yield path


def project_of(path: Path, root: Path = CLAUDE_PROJECTS_DIR) -> str:
    """The project a transcript belongs to.

    Transcripts sit at three observed depths — ``<project>/s.jsonl``,
    ``<project>/<session>/s.jsonl`` and ``<project>/<session>/subagents/agent-*.jsonl`` — so
    the project is the *first* component of the relative path, never a fixed number of
    parents up. Counting parents from the file gives session uuids for the nested cases,
    which then look like distinct projects and silently inflate every per-project statistic.
    """
    return path.relative_to(root).parts[0]


def _iter_records(path: Path, skipped: Counter[str], root: Path) -> Iterator[dict]:
    """Yield assistant text records from one transcript.

    Malformed lines are counted by reason and reported once by the caller, never warned about
    per row: a per-line warning over a 350 MB store buries every other finding.
    """
    project = project_of(path, root)
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped["undecodable JSON line"] += 1
                continue
            if not isinstance(record, dict):
                skipped["JSON line that is not an object"] += 1
                continue
            message = record.get("message")
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                skipped["assistant message with no content list"] += 1
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "text":
                    continue
                text = block.get("text")
                if not isinstance(text, str) or not text.strip():
                    skipped["empty or non-string text block"] += 1
                    continue
                yield {
                    "corpus_id": "claude_sidechain" if record.get("isSidechain") else "claude_main",
                    "part_id": str(record.get("sessionId") or path.stem),
                    "project": project,
                    "model": str(message.get("model") or "unknown"),
                    "timestamp": str(record.get("timestamp") or ""),
                    "transcript": path.name,
                    "text": text,
                }


def harvest(root: Path = CLAUDE_PROJECTS_DIR, *, destination: Path = DOCUMENTS_PARQUET) -> pl.DataFrame:
    """Read the transcript store into one documents table and persist it.

    Returns the table. Rows are sorted by ``(corpus_id, part_id, timestamp, transcript)`` so
    two runs over an unchanged store produce byte-identical output.
    """
    skipped: Counter[str] = Counter()
    rows: list[dict] = []
    transcripts = 0
    for path in iter_transcripts(root):
        transcripts += 1
        rows.extend(_iter_records(path, skipped, root))

    if not rows:
        raise ValueError(
            f"no assistant text found under {root}. The store exists but holds no usable prose; "
            "this is a refusal rather than an empty result, because an empty corpus would "
            "silently produce a comparison against nothing."
        )

    frame = pl.DataFrame(rows).sort("corpus_id", "part_id", "timestamp", "transcript")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(destination)

    for reason, count in sorted(skipped.items()):
        logger.warning("skipped %d records: %s", count, reason)
    logger.info(
        "harvested %d text blocks from %d transcripts at %s",
        frame.height,
        transcripts,
        datetime.now(UTC).isoformat(),
    )
    return frame


def summarise(frame: pl.DataFrame) -> pl.DataFrame:
    """Per-stratum totals: documents, sessions, projects, characters."""
    return (
        frame.group_by("corpus_id")
        .agg(
            pl.len().alias("documents"),
            pl.col("part_id").n_unique().alias("sessions"),
            pl.col("project").n_unique().alias("projects"),
            pl.col("text").str.len_chars().sum().alias("characters"),
        )
        .sort("corpus_id")
    )

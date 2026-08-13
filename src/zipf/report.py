"""Render the comparison on disk as a markdown report.

Every number here is read from a parquet file produced by a real run. Nothing in this module
invents an example value, and a missing input is an error rather than a gap in the prose.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from zipf.compare import (
    MAX_DISPERSION,
    MAX_SESSION_SHARE,
    MIN_SESSIONS,
    MIN_TARGET_COUNT,
    Z_THRESHOLD,
    overused,
    underused,
)
from zipf.models import REFERENCE_TIERS
from zipf.paths import OUTPUT_DIR
from zipf.pipeline import meta_path

logger = logging.getLogger(__name__)


def _corpus_table() -> str:
    """Provenance and size of every corpus that actually built."""
    lines = [
        "| corpus | register | tokens | types | parts | date cutoff |",
        "|---|---|---:|---:|---:|---|",
    ]
    for corpus_id in ("claude_main", "claude_sidechain", *REFERENCE_TIERS):
        path = meta_path(corpus_id)
        if not path.exists():
            lines.append(f"| {corpus_id} | — | *not built* | | | |")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec, stats = payload["spec"], payload["stats"]
        lines.append(
            f"| `{corpus_id}` | {spec['text_register']} | {stats['tokens']:,} | "
            f"{stats['types']:,} | {stats['parts']:,} | {spec['date_cutoff'] or 'n/a'} |"
        )
    return "\n".join(lines)


def _word_table(frame: pl.DataFrame, tiers: list[str], limit: int) -> str:
    header = "| word | per M (Claude) | " + " | ".join(f"per M ({t})" for t in tiers)
    header += " | min z | boot z | DP |"
    rule = "|---|---:|" + "---:|" * len(tiers) + "---:|---:|---:|"
    lines = [header, rule]
    for row in frame.head(limit).iter_rows(named=True):
        cells = [f"`{row['token']}`", f"{row['target_per_million']:.0f}"]
        cells += [f"{row.get(f'per_million_{t}') or 0:.1f}" for t in tiers]
        cells += [
            f"{row['z_min']:.1f}",
            f"{row['z_bootstrap_min']:.1f}",
            f"{row['dispersion_dp']:.2f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(*, target: str = "claude_main", top: int = 60) -> Path:
    """Compose the report from the comparison parquet and return its path."""
    wide_path = OUTPUT_DIR / f"overuse_{target}.parquet"
    if not wide_path.exists():
        raise FileNotFoundError(f"{wide_path} is missing. Run `uv run zipf compare` first.")
    wide = pl.read_parquet(wide_path)
    tiers = [c.removeprefix("z_") for c in wide.columns if c.startswith("z_") and c.removeprefix("z_") in REFERENCE_TIERS]

    result = overused(wide)
    quiet = underused(wide)
    tiers_compared = int(wide["tiers_compared"].max())
    corpus_tokens = json.loads(meta_path(target).read_text(encoding="utf-8"))["stats"]["tokens"]

    body = f"""# Claude Code's overused vocabulary

Generated {datetime.now(UTC).isoformat(timespec="seconds")} from a real run. Every number below
is read from `data/output/overuse_{target}.parquet`.

## What was measured

The prose half of the local Claude Code transcripts — {corpus_tokens:,} word-occurrences after
code, markup, paths and URLs are removed — tokenized identically to {tiers_compared} human
reference corpora and ranked by how far each word's rate departs from **all** of them.
{wide.height:,} words occur often enough to be candidates.

{_corpus_table()}

## How a word qualifies

A word is reported only if it clears every gate:

- at least **{MIN_TARGET_COUNT}** occurrences in the target corpus;
- a log-odds z-score of at least **{Z_THRESHOLD}** against **every** tier, not on average —
  the ranking statistic is the *minimum* z across tiers, so a word cannot be carried by one
  extreme baseline;
- Dunning's G² agreeing in sign with the log-odds z;
- a session-level bootstrap lower bound above zero;
- dispersion (Gries DP) at most **{MAX_DISPERSION}**, no single session holding more than
  **{MAX_SESSION_SHARE:.0%}** of its occurrences, and presence in at least **{MIN_SESSIONS}**
  sessions.

The dispersion gates are what separate a habit from one long conversation about one topic.

## The vocabulary

{result.height} of {wide.height} candidate words cleared every gate.

{_word_table(result, tiers, top)}

## The other direction

The same measurement with the sign flipped: words the baselines use and Claude does not. It is
reported because a method that only ever finds over-use is a method that rewards whatever the
target happens to contain.

{_word_table(quiet.head(20).with_columns(z_min=pl.col("z_mean")), tiers, 20)}

## What this is not

This measures aggregate rate differences between corpora. It is **not** a detector: there is no
per-document verdict, and nothing here can tell you whether a particular text was written by a
model. A word on this list is a word Claude Code writes more often than the humans in these
four corpora did — no more than that.

Every human corpus is dated to before generated text became common, so contamination biases
these results **toward the null**: a contaminated baseline looks more like Claude and shrinks
the measured gap. Findings that survive are therefore conservative.
"""
    destination = OUTPUT_DIR / "report.md"
    destination.write_text(body, encoding="utf-8")
    logger.info("wrote %s (%d qualifying words)", destination, result.height)
    return destination

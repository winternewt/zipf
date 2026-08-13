"""Command line surface.

Pipeline order, each step reading the previous step's output from ``data/``::

    zipf harvest      ~/.claude/projects -> data/interim/claude_documents.parquet
    zipf fetch        reference corpora  -> data/input/<tier>/
    zipf count        text               -> data/output/counts_*.parquet
    zipf compare      counts             -> data/output/overuse_*.parquet
    zipf report       results            -> data/output/report.md
    zipf calibrate    the null test that says whether any of it means anything
"""

from __future__ import annotations

import logging
from typing import Annotated

import polars as pl
import typer
from dotenv import load_dotenv

from zipf import compare as compare_module
from zipf.corpora import TIERS, fetch_tier
from zipf.domain import DOMAIN_THRESHOLD, annotate, empirical_threshold
from zipf.harvest import harvest as harvest_store
from zipf.harvest import summarise
from zipf.models import REFERENCE_TIERS
from zipf import ngrams as ngrams_module
from zipf.nulltest import run_null_test
from zipf.paths import OUTPUT_DIR, ensure_dirs
from zipf.pipeline import (
    load_totals as _load_totals,
    CLAUDE_SPEC,
    DEFAULT_TOKEN_CAP,
    SIDECHAIN_SPEC,
    build_reference,
    count_claude,
    load_totals,
    persist,
)
from zipf.report import write_report

app = typer.Typer(add_completion=False, help=__doc__)


def _setup(verbose: bool = True) -> None:
    """Load ``.env`` and configure logging. Every command starts here."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ensure_dirs()


@app.command()
def harvest() -> None:
    """Read the Claude Code transcript store into a documents table."""
    _setup()
    frame = harvest_store()
    typer.echo(summarise(frame))
    typer.echo(f"\nwrote {frame.height} documents")


@app.command()
def fetch(
    tier: Annotated[str, typer.Option(help="Tier name, or 'all'.")] = "all",
) -> None:
    """Download the reference corpora."""
    _setup()
    names = list(REFERENCE_TIERS) if tier == "all" else [tier]
    for name in names:
        if name not in TIERS:
            raise typer.BadParameter(f"unknown tier {name!r}; known: {sorted(TIERS)}")
        paths = fetch_tier(TIERS[name])
        typer.echo(f"{name}: {len(paths)} files in {TIERS[name].directory}")


@app.command()
def count(
    corpus: Annotated[str, typer.Option(help="Corpus id, or 'all'.")] = "all",
    token_cap: Annotated[int, typer.Option(help="Stop a reference tier at N tokens.")] = DEFAULT_TOKEN_CAP,
) -> None:
    """Tokenize and count. Claude strata first: the reference tiers need their vocabulary."""
    _setup()
    wanted = (
        ["claude_main", "claude_sidechain", *REFERENCE_TIERS] if corpus == "all" else [corpus]
    )

    for stratum, spec in (("claude_main", CLAUDE_SPEC), ("claude_sidechain", SIDECHAIN_SPEC)):
        if stratum in wanted:
            persist(count_claude(stratum), spec)

    reference_names = [n for n in wanted if n in TIERS]
    if not reference_names:
        return

    # The focus vocabulary comes from the target corpus, and only restricts which tokens get a
    # per-part breakdown. Corpus totals stay unfiltered, so no rate is affected by it.
    main_totals, _ = load_totals("claude_main")
    focus = frozenset(main_totals)
    for name in reference_names:
        build_reference(name, focus, token_cap=token_cap)


@app.command()
def compare(
    target: Annotated[str, typer.Option(help="Corpus to rank.")] = "claude_main",
    min_count: Annotated[int, typer.Option(help="Minimum occurrences in the target.")] = compare_module.MIN_TARGET_COUNT,
    draws: Annotated[int, typer.Option(help="Bootstrap resamples over sessions.")] = 500,
    top: Annotated[int, typer.Option(help="Rows to print.")] = 40,
    fold: Annotated[str, typer.Option(help="Morphology: none | nominal | inflection.")] = "none",
) -> None:
    """Rank the target vocabulary against every available reference tier."""
    _setup()
    long, wide = compare_module.build_comparison(
        target=target, min_count=min_count, bootstrap_draws=draws, fold_level=fold
    )
    suffix = "" if fold == "none" else f"_{fold}"
    long.write_parquet(OUTPUT_DIR / f"overuse_long_{target}{suffix}.parquet")
    wide.write_parquet(OUTPUT_DIR / f"overuse_{target}{suffix}.parquet")

    result = compare_module.overused(wide)
    typer.echo(
        f"\n{result.height} words over-used against all {wide['tiers_compared'].max()} tiers "
        f"and well dispersed, of {wide.height} candidates\n"
    )
    with pl.Config(tbl_rows=top, tbl_cols=12, fmt_str_lengths=24):
        typer.echo(
            result.select(
                "token", "target_count", "target_per_million", "z_min", "z_bootstrap_min",
                "dispersion_dp", "sessions_present",
            ).head(top)
        )


@app.command()
def calibrate(
    draws: Annotated[int, typer.Option()] = 200,
) -> None:
    """Null test: compare the target corpus against itself, split by session.

    A method that reports significant over-use between two halves of one corpus is
    miscalibrated, and no unit test would say so. This is the check that decides whether any
    number the pipeline produces is worth quoting.
    """
    _setup()
    run_null_test(draws=draws)


@app.command()
def chains(
    top: Annotated[int, typer.Option(help="Rows to print per chain length.")] = 20,
    token_cap: Annotated[int, typer.Option()] = ngrams_module.DEFAULT_TOKEN_CAP,
) -> None:
    """Extend the comparison to 2/3/4-gram chains.

    Gated on the unigram result: phrase work is another full pass over every corpus and is
    worthless if the single-word comparison shows nothing.
    """
    _setup()
    results = ngrams_module.run(token_cap=token_cap)
    for n, frame in sorted(results.items()):
        qualifying = frame.filter(
            (pl.col("tiers_agreeing") == pl.col("tiers_compared"))
            & (pl.col("dispersion_dp") <= compare_module.MAX_DISPERSION)
            & (pl.col("max_session_share") <= compare_module.MAX_SESSION_SHARE)
            & (pl.col("sessions_present") >= compare_module.MIN_SESSIONS)
        )
        typer.echo(f"\n=== {n}-grams: {qualifying.height} qualifying of {frame.height} ===")
        with pl.Config(tbl_rows=top, tbl_cols=8, fmt_str_lengths=40):
            typer.echo(
                qualifying.select(
                    "ngram", "target_count", "target_per_million", "z_min",
                    "dispersion_dp", "sessions_present",
                ).head(top)
            )


@app.command()
def domain(
    target: Annotated[str, typer.Option()] = "claude_main",
    fold: Annotated[str, typer.Option(help="Morphology level used for the comparison.")] = "none",
    false_positive_rate: Annotated[float, typer.Option(help="Target FPR for the threshold.")] = 0.01,
    top: Annotated[int, typer.Option()] = 30,
) -> None:
    """Separate domain vocabulary from style, and recalibrate the significance threshold.

    Two corrections, for two different kinds of damage. Domain vocabulary occupies the ranking,
    which a topic-matched baseline fixes. It also widens the spread of log-odds, which means a
    fixed z threshold does not mean what it would in a topic-matched comparison — that is fixed
    by reading the threshold off the corpus's own null distribution instead of assuming one.
    """
    _setup()
    suffix = "" if fold == "none" else f"_{fold}"
    path = OUTPUT_DIR / f"overuse_{target}{suffix}.parquet"
    if not path.exists():
        raise typer.BadParameter(f"{path} is missing; run `uv run zipf compare --fold {fold}` first")

    wide = pl.read_parquet(path)
    tier_totals = {}
    for tier in REFERENCE_TIERS:
        try:
            tier_totals[tier] = _load_totals(tier)
        except FileNotFoundError:
            typer.echo(f"tier {tier} missing; specialisation will use what is available")
    annotated = annotate(wide, tier_totals, stratum=target)

    null = run_null_test(draws=200)
    threshold = empirical_threshold(null["z"].to_numpy(), false_positive_rate=false_positive_rate)

    annotated = annotated.with_columns(
        clears_empirical=(pl.col("z_min") >= threshold),
    )
    annotated.write_parquet(OUTPUT_DIR / f"overuse_annotated_{target}{suffix}.parquet")

    passes = (pl.col("tiers_agreeing") == pl.col("tiers_compared")) & pl.col("well_dispersed")
    qualifying = annotated.filter(passes)
    recalibrated = qualifying.filter(pl.col("clears_empirical"))
    style = recalibrated.filter(~pl.col("is_domain") & ~pl.col("is_version_control"))

    typer.echo(
        f"\nempirical threshold at FPR {false_positive_rate:.1%}: z >= {threshold:.2f} "
        f"(the assumed constant was {compare_module.Z_THRESHOLD:.2f})\n"
        f"  qualifying at the assumed threshold : {qualifying.height}\n"
        f"  qualifying at the empirical threshold: {recalibrated.height}\n"
        f"  of those, domain vocabulary (specialisation >= {DOMAIN_THRESHOLD}): "
        f"{recalibrated.filter(pl.col('is_domain')).height}\n"
        f"  of those, version-control vocabulary  : "
        f"{recalibrated.filter(pl.col('is_version_control')).height}\n"
        f"  of those, style vocabulary            : {style.height}\n"
    )
    with pl.Config(tbl_rows=top, tbl_cols=10, fmt_str_lengths=26):
        typer.echo(
            style.sort(["z_min", "token"], descending=[True, False])
            .select(
                "token", "target_per_million", "z_min", "specialisation", "vcs_per_million",
                "project_dp", "sessions_present",
            )
            .head(top)
        )


@app.command()
def report(
    target: Annotated[str, typer.Option()] = "claude_main",
    top: Annotated[int, typer.Option()] = 60,
) -> None:
    """Write the markdown report from the comparison already on disk."""
    _setup()
    path = write_report(target=target, top=top)
    typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()

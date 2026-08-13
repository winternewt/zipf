"""Rank the target corpus's vocabulary by how far it departs from the human baselines.

The ranking statistic is the **minimum** z-score across the reference tiers, not the mean.
A mean lets one extreme tier carry a word: `commit` scores enormously against Gutenberg and
not at all against StackOverflow, and averaging would promote it. Taking the minimum states
the claim the project actually wants to make — *over-used against every human baseline we
tried, including the one that shares its subject matter*.

Three gates run before ranking, and a word must clear all of them:

1. **Frequency.** Below a floor there is not enough evidence for any statistic to mean much.
2. **Agreement.** The log-odds z and Dunning's G² must agree in sign. They rest on different
   assumptions; where they disagree, neither is reported.
3. **Dispersion.** The word must be spread across sessions rather than concentrated in one.
   This is the gate that separates a habit from a single long conversation about one topic.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from zipf.models import REFERENCE_TIERS
from zipf.morphology import apply_fold, build_fold_map
from zipf.pipeline import load_part_matrix, load_totals
from zipf.stats import (
    DEFAULT_PRIOR_MASS,
    bootstrap_z,
    dispersion_excess,
    gries_dp,
    log_likelihood_g2,
    log_odds_dirichlet,
    occupancy,
)

logger = logging.getLogger(__name__)

#: A word must occur at least this often in the target corpus to be a candidate. Below this,
#: the Dirichlet prior dominates and the z-score mostly reflects the background, not the data.
MIN_TARGET_COUNT = 20

#: Dispersion ceiling. Gries DP above this means the occurrences are concentrated enough that
#: the word is more plausibly one session's topic than a habit.
MAX_DISPERSION = 0.75

#: How much more concentrated than words of the same frequency a word may be. This is the
#: frequency-neutral counterpart of MAX_DISPERSION; see stats.dispersion_excess for why the
#: flat ceiling alone is not frequency-neutral.
MAX_DISPERSION_EXCESS = 0.05

#: No single session may account for more than this share of a word's occurrences.
MAX_SESSION_SHARE = 0.35

#: Minimum sessions a word must appear in at all.
MIN_SESSIONS = 8

#: z-score a word must clear against a tier to count that tier as agreeing.
Z_THRESHOLD = 3.0


def build_comparison(
    target: str = "claude_main",
    tiers: tuple[str, ...] = REFERENCE_TIERS,
    *,
    prior_mass: float = DEFAULT_PRIOR_MASS,
    min_count: int = MIN_TARGET_COUNT,
    bootstrap_draws: int = 500,
    fold_level: str = "none",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compare the target corpus against each available tier.

    Returns ``(long, wide)``: one row per (token, tier), and one row per token with the
    per-tier z-scores pivoted out plus the gates and the ranking statistic.

    A tier whose counts are missing is reported as **unknown** rather than skipped silently:
    it is dropped from the tier list, named in the log, and the resulting `tiers_compared`
    column says how many baselines actually ran. Three-of-four is a legitimate result; three
    presented as four is a false one.
    """
    target_totals, target_size = load_totals(target)

    available: list[str] = []
    tier_totals: dict[str, tuple[dict[str, int], int]] = {}
    for tier in tiers:
        try:
            tier_totals[tier] = load_totals(tier)
            available.append(tier)
        except FileNotFoundError:
            logger.warning("tier %s has no counts on disk; it will be reported as unknown", tier)
    if not available:
        raise FileNotFoundError(
            "no reference tiers are available, so there is nothing to compare against. "
            "Refusing rather than emitting a table of unknowns."
        )

    # Morphological folding, if asked for. The map is built from the UNION of every corpus's
    # vocabulary, never the target's alone: folding the target's forms while leaving the
    # baselines' forms split would inflate the target's rate for exactly the words under test.
    fold: dict[str, str] = {}
    if fold_level != "none":
        pooled: dict[str, int] = dict(target_totals)
        for tier in available:
            for token, count in tier_totals[tier][0].items():
                pooled[token] = pooled.get(token, 0) + count
        fold, _ = build_fold_map(pooled, level=fold_level)
        target_totals = apply_fold(target_totals, fold)
        tier_totals = {
            name: (apply_fold(totals, fold), size) for name, (totals, size) in tier_totals.items()
        }

    vocabulary = sorted(t for t, c in target_totals.items() if c >= min_count)
    if not vocabulary:
        raise ValueError(f"no token reaches the minimum count of {min_count}")
    logger.info(
        "%d candidate types (of %d) at count >= %d; comparing against %s",
        len(vocabulary),
        len(target_totals),
        min_count,
        ", ".join(available),
    )

    target_counts = np.array([target_totals[t] for t in vocabulary], dtype=np.float64)

    # The prior's shape is the pooled background across the target and every available tier.
    # Pooling matters: a prior taken from the target alone would be the target checking itself.
    background = target_counts.copy()
    # The denominator is every corpus's FULL token count, not the sum over candidates. See
    # stats.log_odds_dirichlet: renormalising over the candidate subset inflates the prior.
    background_total = float(target_size)
    for tier in available:
        totals, tier_size = tier_totals[tier]
        background += np.array([totals.get(t, 0) for t in vocabulary], dtype=np.float64)
        background_total += float(tier_size)

    matrix, part_sizes, part_ids = load_part_matrix(target, vocabulary, fold=fold)
    dispersion = gries_dp(matrix, part_sizes)
    sessions_present, max_share = occupancy(matrix)

    rows: list[pl.DataFrame] = []
    for tier in available:
        totals, tier_size = tier_totals[tier]
        reference_counts = np.array([totals.get(t, 0) for t in vocabulary], dtype=np.float64)
        delta, z = log_odds_dirichlet(
            target_counts,
            reference_counts,
            background,
            prior_mass=prior_mass,
            target_total=target_size,
            reference_total=tier_size,
            background_total=background_total,
        )
        g2 = log_likelihood_g2(
            target_counts,
            reference_counts,
            target_total=target_size,
            reference_total=tier_size,
        )
        z_low = bootstrap_z(
            matrix,
            part_sizes,
            reference_counts,
            background,
            draws=bootstrap_draws,
            prior_mass=prior_mass,
            reference_total=tier_size,
            background_total=background_total,
        )
        rows.append(
            pl.DataFrame(
                {
                    "token": vocabulary,
                    "reference": [tier] * len(vocabulary),
                    "target_count": target_counts.astype(np.int64),
                    "reference_count": reference_counts.astype(np.int64),
                    "target_per_million": target_counts / target_size * 1e6,
                    "reference_per_million": reference_counts / tier_size * 1e6,
                    "log_odds": delta,
                    "z": z,
                    "g2": g2,
                    "z_bootstrap_low": z_low,
                }
            )
        )

    long = pl.concat(rows).with_columns(
        agrees=(pl.col("z") >= Z_THRESHOLD)
        & (pl.col("g2") > 0)
        & (pl.col("z_bootstrap_low") > 0)
    )

    expected_dp, excess_dp = dispersion_excess(dispersion, target_counts)
    dispersion_frame = pl.DataFrame(
        {
            "token": vocabulary,
            "dispersion_dp": dispersion,
            "dispersion_expected": expected_dp,
            "dispersion_excess": excess_dp,
            "sessions_present": sessions_present.astype(np.int64),
            "max_session_share": max_share,
        }
    )

    wide = (
        long.group_by("token")
        .agg(
            pl.col("target_count").first(),
            pl.col("target_per_million").first(),
            pl.col("z").min().alias("z_min"),
            pl.col("z").mean().alias("z_mean"),
            pl.col("z_bootstrap_low").min().alias("z_bootstrap_min"),
            pl.col("agrees").sum().alias("tiers_agreeing"),
            pl.len().alias("tiers_compared"),
        )
        .join(dispersion_frame, on="token", how="left")
        .join(
            long.pivot(on="reference", index="token", values="z").rename(
                {t: f"z_{t}" for t in available}
            ),
            on="token",
            how="left",
        )
        .join(
            long.pivot(on="reference", index="token", values="reference_per_million").rename(
                {t: f"per_million_{t}" for t in available}
            ),
            on="token",
            how="left",
        )
        .with_columns(
            # The flat gate. Published as the primary result because it is the stricter of the
            # two and was fixed before any word was looked at.
            well_dispersed=(pl.col("dispersion_dp") <= MAX_DISPERSION)
            & (pl.col("max_session_share") <= MAX_SESSION_SHARE)
            & (pl.col("sessions_present") >= MIN_SESSIONS),
            # The frequency-neutral gate: is this word more concentrated than words of its own
            # frequency? Reported alongside rather than instead of, because the flat gate's
            # bias was discovered after the first ranking existed, and quietly swapping to the
            # gate that admits more words is indistinguishable from tuning for a nicer result.
            well_dispersed_conditional=(pl.col("dispersion_excess") <= MAX_DISPERSION_EXCESS)
            & (pl.col("max_session_share") <= MAX_SESSION_SHARE)
            & (pl.col("sessions_present") >= MIN_SESSIONS),
        )
        # Ties break on the token so two runs rank identically.
        .sort(["z_min", "token"], descending=[True, False])
    )
    return long, wide


def overused(wide: pl.DataFrame, *, require_all_tiers: bool = True) -> pl.DataFrame:
    """The words that cleared every gate, most over-used first."""
    frame = wide.filter(pl.col("well_dispersed"))
    if require_all_tiers:
        frame = frame.filter(pl.col("tiers_agreeing") == pl.col("tiers_compared"))
    else:
        frame = frame.filter(pl.col("tiers_agreeing") > 0)
    return frame.sort(["z_min", "token"], descending=[True, False])


def underused(wide: pl.DataFrame) -> pl.DataFrame:
    """Words the target uses far *less* than every baseline.

    Reported because it is the same measurement with the sign flipped, and because it is the
    cheapest available check that the method is not simply rewarding whatever the target
    happens to contain.
    """
    return wide.sort(["z_mean", "token"], descending=[False, False])

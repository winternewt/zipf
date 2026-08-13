"""The null test: does the method find nothing when there is nothing to find?

The target corpus is split into two halves **by session**, and one half is compared against
the other with exactly the pipeline used for the real comparison. Both halves are the same
author, the same models, the same register and largely the same topics, so the honest answer
is "no words are over-used".

Whatever this test reports is the method's false-positive floor. If it returns a long list,
every number the pipeline produces is worth exactly nothing, and no unit test would have said
so — the code would be working correctly and the claim would still be wrong.

Splitting is by session rather than by document because documents inside a session share a
topic. A document-level split would leak the same conversation into both halves, and the two
halves would look far more similar than two genuinely independent samples, which would make
the test pass for the wrong reason.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from zipf.compare import (
    MAX_DISPERSION,
    MAX_SESSION_SHARE,
    MIN_SESSIONS,
    MIN_TARGET_COUNT,
    Z_THRESHOLD,
)
from zipf.counts import CorpusCounts
from zipf.harvest import DOCUMENTS_PARQUET
from zipf.stats import DEFAULT_PRIOR_MASS, bootstrap_z, gries_dp, log_likelihood_g2, log_odds_dirichlet, occupancy

logger = logging.getLogger(__name__)


def split_sessions(session_ids: list[str], *, seed: int = 20260813) -> tuple[set[str], set[str]]:
    """Deterministically halve a list of session ids."""
    rng = np.random.default_rng(seed)
    ordered = sorted(set(session_ids))
    shuffled = list(rng.permutation(ordered))
    midpoint = len(shuffled) // 2
    return set(shuffled[:midpoint]), set(shuffled[midpoint:])


def run_null_test(
    *,
    stratum: str = "claude_main",
    draws: int = 200,
    min_count: int = MIN_TARGET_COUNT,
    prior_mass: float = DEFAULT_PRIOR_MASS,
    seed: int = 20260813,
) -> pl.DataFrame:
    """Compare one half of the target corpus against the other and report what survives."""
    frame = pl.read_parquet(DOCUMENTS_PARQUET).filter(pl.col("corpus_id") == stratum)
    left_ids, right_ids = split_sessions(frame["part_id"].to_list(), seed=seed)

    left = CorpusCounts("null_left")
    right = CorpusCounts("null_right")
    for part_id, text in zip(frame["part_id"], frame["text"], strict=True):
        target = left if str(part_id) in left_ids else right
        target.add(str(part_id), str(text), preprocessor="markdown")

    logger.info(
        "null split: %d sessions / %d tokens vs %d sessions / %d tokens",
        left.parts,
        left.tokens,
        right.parts,
        right.tokens,
    )

    vocabulary = sorted(t for t, c in left.totals.items() if c >= min_count)
    left_counts = np.array([left.totals[t] for t in vocabulary], dtype=np.float64)
    right_counts = np.array([right.totals.get(t, 0) for t in vocabulary], dtype=np.float64)
    background = left_counts + right_counts

    part_ids = sorted(left.per_part)
    matrix = np.zeros((len(part_ids), len(vocabulary)), dtype=np.float64)
    vocab_index = {t: i for i, t in enumerate(vocabulary)}
    for row, part_id in enumerate(part_ids):
        for token, value in left.per_part[part_id].items():
            column = vocab_index.get(token)
            if column is not None:
                matrix[row, column] = value
    sizes = np.array([left.part_sizes[p] for p in part_ids], dtype=np.float64)

    _, z = log_odds_dirichlet(
        left_counts,
        right_counts,
        background,
        prior_mass=prior_mass,
        target_total=left.tokens,
        reference_total=right.tokens,
    )
    g2 = log_likelihood_g2(
        left_counts, right_counts, target_total=left.tokens, reference_total=right.tokens
    )
    z_low = bootstrap_z(
        matrix,
        sizes,
        right_counts,
        background,
        draws=draws,
        prior_mass=prior_mass,
        reference_total=right.tokens,
    )
    dispersion = gries_dp(matrix, sizes)
    sessions_present, max_share = occupancy(matrix)

    result = pl.DataFrame(
        {
            "token": vocabulary,
            "left_count": left_counts.astype(np.int64),
            "right_count": right_counts.astype(np.int64),
            "z": z,
            "g2": g2,
            "z_bootstrap_low": z_low,
            "dispersion_dp": dispersion,
            "sessions_present": sessions_present.astype(np.int64),
            "max_session_share": max_share,
        }
    ).with_columns(
        survives=(pl.col("z") >= Z_THRESHOLD)
        & (pl.col("g2") > 0)
        & (pl.col("z_bootstrap_low") > 0)
        & (pl.col("dispersion_dp") <= MAX_DISPERSION)
        & (pl.col("max_session_share") <= MAX_SESSION_SHARE)
        & (pl.col("sessions_present") >= MIN_SESSIONS)
    )

    survivors = result.filter(pl.col("survives")).sort(["z", "token"], descending=[True, False])
    logger.info(
        "null test: %d of %d candidate words pass every gate (false-positive floor %.2f%%)",
        survivors.height,
        result.height,
        100 * survivors.height / max(result.height, 1),
    )
    return result

"""The statistics that turn two frequency tables into a defensible claim.

Why not a frequency ratio: a ratio carries no variance, so a word seen three times in the
target and once in the baseline scores 3x and outranks a word seen thirty thousand times
against ten thousand. Ratios put hapax noise at the top of the list.

Four numbers are computed for every candidate word, and a word has to survive all four to be
reported:

:func:`log_odds_dirichlet`
    Monroe, Colaresi & Quinn (2008), "Fightin' Words". A log-odds difference regularised by an
    informative Dirichlet prior taken from the pooled background, divided by its own standard
    error. The prior shrinks rare words toward the background, which is exactly the behaviour
    a raw ratio lacks. This is the headline statistic.
:func:`log_likelihood_g2`
    Dunning's G², the standard corpus-linguistics keyness test. Included as an independent
    cross-check with different assumptions, not as a second opinion from the same method: if
    G² and the log-odds z disagree on a word, that word is not reported.
:func:`gries_dp`
    Gries' Deviation of Proportions. Answers "is this word spread across the corpus, or did one
    long session produce all of it?" A tic is spread; an artifact is concentrated.
:func:`bootstrap_z`
    Resamples *sessions* with replacement, not tokens. Tokens within a session are not
    independent — an author who says "gap" once says it again three lines later — so a
    token-level interval would be far too narrow. The reported bound is the pessimistic
    percentile across resamples.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Default prior mass for the Dirichlet, in pseudo-tokens spread over the background shape.
#: 5000 against a ~430k-token target is roughly 1% shrinkage: enough to tame hapaxes, small
#: enough not to flatten a real effect. Sensitivity to this choice is reported by the CLI.
DEFAULT_PRIOR_MASS = 5000.0


def log_odds_dirichlet(
    target: np.ndarray,
    reference: np.ndarray,
    background: np.ndarray,
    *,
    prior_mass: float = DEFAULT_PRIOR_MASS,
    target_total: float | None = None,
    reference_total: float | None = None,
    background_total: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Monroe log-odds ratio with an informative Dirichlet prior, and its z-score.

    All three arrays are counts over the *same* vocabulary in the *same* order.

    ``background`` supplies only the *shape* of the prior and is pooled across every corpus.
    Deriving it from the target alone would make the comparison check the target against
    itself, and it would agree.

    ``target_total`` and ``reference_total`` are the corpora's **full** token counts. They
    must be passed whenever the count arrays have been restricted to a candidate vocabulary,
    which is the normal case: a reference corpus is sliced down to the words the target
    actually uses, and summing the slice would understate its size by an order of magnitude
    and inflate every reference rate to match. They default to the array sums, which is
    correct only for an unrestricted vocabulary.

    Returns ``(delta, z)`` where ``delta`` is the regularised log-odds difference (positive =
    the target uses the word more) and ``z`` is ``delta`` over its standard error.
    """
    target = target.astype(np.float64)
    reference = reference.astype(np.float64)
    background = background.astype(np.float64)

    # The prior must be the background *rate* in the full corpora, not the background's shape
    # renormalised over whichever words happen to be candidates. Renormalising over a subset
    # inflates every alpha by (all tokens / candidate tokens), and that factor explodes as the
    # candidate set narrows: with 347 candidate 4-grams the pseudo-counts reached the hundreds
    # and swamped the real ones, scoring an ordinary English phrase at a 1.2x rate ratio as
    # z=75. Pass background_total whenever the arrays are a candidate subset.
    denominator = background.sum() if background_total is None else float(background_total)
    if denominator <= 0:
        raise ValueError("background counts are empty; the prior would be undefined")
    alpha = prior_mass * (background / denominator)
    # alpha_total is the prior mass actually landing on this vocabulary, which is less than
    # prior_mass whenever the vocabulary is a subset. Using prior_mass here instead would put
    # mass on words that are not in the comparison.
    alpha_total = float(alpha.sum())

    n_target = float(target.sum()) if target_total is None else float(target_total)
    n_reference = float(reference.sum()) if reference_total is None else float(reference_total)

    target_adj = target + alpha
    reference_adj = reference + alpha
    # The complement term uses the corpus total plus total prior mass, per the paper: the
    # odds are word-against-everything-else, not word-against-word.
    target_rest = n_target + alpha_total - target_adj
    reference_rest = n_reference + alpha_total - reference_adj

    if np.any(target_rest <= 0) or np.any(reference_rest <= 0):
        raise ValueError(
            "a word's count exceeds its corpus total; target and reference vocabularies are "
            "misaligned"
        )

    delta = np.log(target_adj / target_rest) - np.log(reference_adj / reference_rest)
    variance = 1.0 / target_adj + 1.0 / reference_adj
    return delta, delta / np.sqrt(variance)


def log_likelihood_g2(
    target: np.ndarray,
    reference: np.ndarray,
    *,
    target_total: float | None = None,
    reference_total: float | None = None,
) -> np.ndarray:
    """Dunning's G², signed so positive means the target over-uses the word.

    Zero-count terms contribute zero, which is the limit of ``y*log(y/E)`` as ``y -> 0``.

    As in :func:`log_odds_dirichlet`, pass the full corpus totals whenever the arrays are
    restricted to a candidate vocabulary.
    """
    target = target.astype(np.float64)
    reference = reference.astype(np.float64)
    n_target = target.sum() if target_total is None else float(target_total)
    n_reference = reference.sum() if reference_total is None else float(reference_total)
    pooled = target + reference
    total = n_target + n_reference

    expected_target = n_target * pooled / total
    expected_reference = n_reference * pooled / total

    with np.errstate(divide="ignore", invalid="ignore"):
        term_target = np.where(target > 0, target * np.log(target / expected_target), 0.0)
        term_reference = np.where(
            reference > 0, reference * np.log(reference / expected_reference), 0.0
        )
    g2 = 2.0 * (term_target + term_reference)
    direction = np.sign(target / n_target - reference / n_reference)
    return g2 * direction


def gries_dp(part_counts: np.ndarray, part_sizes: np.ndarray) -> np.ndarray:
    """Gries' Deviation of Proportions, normalised to [0, 1].

    ``part_counts`` is ``(parts, vocab)``; ``part_sizes`` is ``(parts,)``.

    0 means the word is distributed exactly in proportion to part sizes — perfectly even.
    1 means every occurrence sits in one part. A word with a strong z-score and a DP near 1 is
    a single session talking about one thing, not a habit.

    Words with zero total count get ``nan``: "no occurrences" has no dispersion, and returning
    0 would falsely read as "perfectly even".
    """
    part_counts = part_counts.astype(np.float64)
    part_sizes = part_sizes.astype(np.float64)
    total_size = part_sizes.sum()
    if total_size <= 0:
        raise ValueError("part sizes sum to zero")

    expected = part_sizes / total_size
    totals = part_counts.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        observed = np.where(totals > 0, part_counts / totals, np.nan)
    raw = 0.5 * np.nansum(np.abs(observed - expected[:, None]), axis=0)
    # The theoretical maximum is 1 - min(expected); normalising makes DP comparable across
    # corpora with different part-size distributions.
    ceiling = 1.0 - expected.min()
    dp = raw / ceiling if ceiling > 0 else raw
    return np.where(totals > 0, dp, np.nan)


def dispersion_excess(
    dp: np.ndarray, counts: np.ndarray, *, bins: int = 20, min_per_bin: int = 25
) -> tuple[np.ndarray, np.ndarray]:
    """How concentrated a word is *relative to words of the same frequency*.

    A flat dispersion ceiling is not the frequency-neutral gate it appears to be. Measured on
    this corpus, Spearman's rho between log count and DP is **-0.768**: a 0.75 ceiling rejects
    72% of words occurring 20-30 times and 0% of words occurring over 300 times. A rare word
    *cannot* spread evenly over a hundred sessions however habitual it is, so the flat ceiling
    silently does frequency filtering under a dispersion name, and it discards exactly the
    distinctive low-frequency vocabulary the project exists to find.

    This returns ``(expected, excess)`` where ``expected`` is the median DP of words in the
    same log-count band and ``excess = dp - expected``. Positive means more concentrated than
    its frequency predicts; that is the quantity a dispersion gate should have been testing.

    Bands are quantiles of log count, merged upward until each holds ``min_per_bin`` words, so
    the expectation is not itself estimated from a handful of points.
    """
    dp = np.asarray(dp, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    expected = np.full(dp.shape, np.nan)

    usable = np.isfinite(dp) & (counts > 0)
    if usable.sum() < min_per_bin:
        # Too few words to estimate a frequency-conditional expectation; returning all-nan is
        # the honest answer. A zero here would read as "no word deviates", which is a claim.
        return expected, expected.copy()

    log_counts = np.log(counts[usable])
    order = np.argsort(log_counts, kind="stable")
    indices = np.flatnonzero(usable)[order]
    n_bins = max(1, min(bins, indices.size // min_per_bin))
    for chunk in np.array_split(indices, n_bins):
        if chunk.size:
            expected[chunk] = np.median(dp[chunk])
    return expected, dp - expected


def occupancy(part_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """How many parts contain the word, and the largest share any one part holds.

    Reported beside DP because they fail differently: a word can be in many parts and still be
    80% one session.
    """
    present = (part_counts > 0).sum(axis=0)
    totals = part_counts.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(totals > 0, part_counts.max(axis=0) / totals, np.nan)
    return present, share


def bootstrap_z(
    part_counts: np.ndarray,
    part_sizes: np.ndarray,
    reference: np.ndarray,
    background: np.ndarray,
    *,
    draws: int = 1000,
    prior_mass: float = DEFAULT_PRIOR_MASS,
    percentile: float = 5.0,
    seed: int = 20260813,
    reference_total: float | None = None,
    background_total: float | None = None,
) -> np.ndarray:
    """Resample sessions with replacement and return a pessimistic percentile of the z-score.

    Resampling is over **parts, not tokens**. Tokens inside one session are heavily correlated,
    so a token-level bootstrap would report intervals several times too narrow and would
    certify single-session artifacts.

    The returned value is the ``percentile``-th percentile of the resampled z for each word:
    for a word claimed to be over-used, this is the weakest evidence the resampling produced.
    """
    rng = np.random.default_rng(seed)
    n_parts = part_counts.shape[0]
    if n_parts < 2:
        raise ValueError("bootstrap needs at least two parts")

    # Multinomial resample weights: draws x parts. Equivalent to sampling parts with
    # replacement, but expressible as one matrix product against the part-count matrix.
    weights = rng.multinomial(n_parts, np.full(n_parts, 1.0 / n_parts), size=draws).astype(
        np.float64
    )
    resampled_counts = weights @ part_counts.astype(np.float64)
    resampled_totals = weights @ part_sizes.astype(np.float64)

    zs = np.empty((draws, part_counts.shape[1]), dtype=np.float64)
    for i in range(draws):
        scale = resampled_totals[i]
        if scale <= 0:
            zs[i] = np.nan
            continue
        _, z = log_odds_dirichlet(
            resampled_counts[i],
            reference,
            background,
            prior_mass=prior_mass,
            target_total=scale,
            reference_total=reference_total,
            background_total=background_total,
        )
        zs[i] = z
    return np.nanpercentile(zs, percentile, axis=0)


def zipf_fit(counts: np.ndarray) -> tuple[float, float]:
    """Least-squares slope and intercept of log-frequency against log-rank.

    A Zipfian corpus has a slope near -1. It is reported as a sanity check on the corpus, not
    as evidence about any word: a corpus whose slope is far from -1 has usually been filtered,
    deduplicated or truncated in a way that also distorts the comparison.
    """
    ordered = np.sort(counts[counts > 0])[::-1].astype(np.float64)
    if ordered.size < 10:
        raise ValueError("too few non-zero counts to fit")
    ranks = np.arange(1, ordered.size + 1, dtype=np.float64)
    slope, intercept = np.polyfit(np.log(ranks), np.log(ordered), 1)
    return float(slope), float(intercept)

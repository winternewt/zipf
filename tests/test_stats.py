"""The statistics, checked against closed-form values computed by hand.

The reference numbers below are substitutions into the published formulae, not output
captured from this implementation. Asserting against a second copy of the code would test
that the code equals itself.

Worked case used throughout, a two-word vocabulary::

    target    = [10, 90]      n_target    = 100
    reference = [ 5, 95]      n_reference = 100
    background= [15, 185]     prior_mass  = 10

Monroe's prior is ``alpha = prior_mass * background / sum(background)`` = ``[0.75, 9.25]``,
so ``alpha_total = 10``. For the first word::

    delta = log(10.75 / (100 + 10 - 10.75)) - log(5.75 / (100 + 10 - 5.75))
          = log(10.75 / 99.25) - log(5.75 / 104.25)
          = -2.22284 - (-2.89770) = 0.67486
    var   = 1/10.75 + 1/5.75 = 0.0930233 + 0.1739130 = 0.2669363
    z     = 0.67486 / sqrt(0.2669363) = 1.30619
"""

from __future__ import annotations

import numpy as np
import pytest

from zipf.stats import (
    bootstrap_z,
    dispersion_excess,
    gries_dp,
    log_likelihood_g2,
    log_odds_dirichlet,
    occupancy,
    zipf_fit,
)

TARGET = np.array([10.0, 90.0])
REFERENCE = np.array([5.0, 95.0])
BACKGROUND = np.array([15.0, 185.0])
PRIOR_MASS = 10.0

EXPECTED_DELTA_FIRST = 0.67486
EXPECTED_Z_FIRST = 1.30619
# G2 for the same table: 2 * sum(O * ln(O/E)) with E from the pooled margins.
EXPECTED_G2_FIRST = 1.69899


def test_log_odds_matches_hand_computed_value() -> None:
    delta, z = log_odds_dirichlet(TARGET, REFERENCE, BACKGROUND, prior_mass=PRIOR_MASS)
    assert delta[0] == pytest.approx(EXPECTED_DELTA_FIRST, abs=1e-5)
    assert z[0] == pytest.approx(EXPECTED_Z_FIRST, abs=1e-5)


def test_log_odds_is_antisymmetric() -> None:
    """Swapping target and reference must negate delta exactly."""
    forward, _ = log_odds_dirichlet(TARGET, REFERENCE, BACKGROUND, prior_mass=PRIOR_MASS)
    backward, _ = log_odds_dirichlet(REFERENCE, TARGET, BACKGROUND, prior_mass=PRIOR_MASS)
    assert forward == pytest.approx(-backward)


def test_log_odds_of_identical_corpora_is_zero() -> None:
    delta, z = log_odds_dirichlet(TARGET, TARGET, BACKGROUND, prior_mass=PRIOR_MASS)
    assert delta == pytest.approx(np.zeros_like(delta))
    assert z == pytest.approx(np.zeros_like(z))


def test_explicit_totals_change_the_result() -> None:
    """A restricted reference slice with the true total must not equal the naive version.

    This is the failure the totals arguments exist to prevent: slicing a reference corpus down
    to the target's vocabulary and then summing the slice understates the corpus by orders of
    magnitude, which inflates every reference rate to match the target's.
    """
    naive, _ = log_odds_dirichlet(TARGET, REFERENCE, BACKGROUND, prior_mass=PRIOR_MASS)
    corrected, _ = log_odds_dirichlet(
        TARGET,
        REFERENCE,
        BACKGROUND,
        prior_mass=PRIOR_MASS,
        reference_total=100_000.0,
    )
    # The reference corpus is 1000x larger than the slice suggests, so the same 5 occurrences
    # represent a far rarer word, and the target must look more over-using, not less.
    assert corrected[0] > naive[0]


def test_prior_shrinks_rare_words_more_than_common_ones() -> None:
    """The point of the informative prior: hapaxes must not outrank well-attested words."""
    rare_target = np.array([3.0, 997.0])
    rare_reference = np.array([1.0, 999.0])
    common_target = np.array([300.0, 700.0])
    common_reference = np.array([100.0, 900.0])
    background = np.array([404.0, 3596.0])

    _, rare_z = log_odds_dirichlet(rare_target, rare_reference, background, prior_mass=100.0)
    _, common_z = log_odds_dirichlet(
        common_target, common_reference, background, prior_mass=100.0
    )
    # Both have a 3x-ish rate ratio; only the well-attested one should carry a large z.
    assert common_z[0] > rare_z[0]


def test_empty_background_is_refused() -> None:
    with pytest.raises(ValueError, match="background"):
        log_odds_dirichlet(TARGET, REFERENCE, np.zeros(2), prior_mass=PRIOR_MASS)


def test_g2_matches_hand_computed_value() -> None:
    g2 = log_likelihood_g2(TARGET, REFERENCE)
    assert abs(g2[0]) == pytest.approx(EXPECTED_G2_FIRST, abs=1e-5)


def test_g2_sign_follows_direction() -> None:
    g2 = log_likelihood_g2(TARGET, REFERENCE)
    assert g2[0] > 0  # target uses word 0 more
    assert g2[1] < 0  # and word 1 less


def test_g2_of_identical_distributions_is_zero() -> None:
    assert log_likelihood_g2(TARGET, TARGET) == pytest.approx(np.zeros(2))


def test_g2_handles_zero_counts_without_nan() -> None:
    g2 = log_likelihood_g2(np.array([5.0, 0.0]), np.array([0.0, 5.0]))
    assert np.all(np.isfinite(g2))


def test_dispersion_of_perfectly_even_word_is_zero() -> None:
    """A word occurring in exact proportion to part size has DP 0."""
    part_counts = np.array([[10.0], [20.0], [30.0]])
    part_sizes = np.array([100.0, 200.0, 300.0])
    assert gries_dp(part_counts, part_sizes)[0] == pytest.approx(0.0, abs=1e-12)


def test_dispersion_of_fully_concentrated_word_is_one() -> None:
    """All occurrences in one part gives the normalised maximum."""
    part_counts = np.array([[60.0], [0.0], [0.0]])
    part_sizes = np.array([200.0, 200.0, 200.0])
    assert gries_dp(part_counts, part_sizes)[0] == pytest.approx(1.0)


def test_dispersion_ordering_is_monotone() -> None:
    part_sizes = np.array([100.0, 100.0, 100.0])
    even = np.array([[10.0], [10.0], [10.0]])
    skewed = np.array([[20.0], [8.0], [2.0]])
    concentrated = np.array([[30.0], [0.0], [0.0]])
    values = [gries_dp(m, part_sizes)[0] for m in (even, skewed, concentrated)]
    assert values == sorted(values)


def test_absent_word_has_unknown_dispersion_not_zero() -> None:
    """`nan`, not 0: a word with no occurrences is unknown, not perfectly even."""
    dp = gries_dp(np.array([[0.0], [0.0]]), np.array([50.0, 50.0]))
    assert np.isnan(dp[0])


def test_occupancy_reports_presence_and_largest_share() -> None:
    part_counts = np.array([[8.0, 1.0], [2.0, 0.0], [0.0, 0.0]])
    present, share = occupancy(part_counts)
    assert present.tolist() == [2, 1]
    assert share[0] == pytest.approx(0.8)
    assert share[1] == pytest.approx(1.0)


def test_bootstrap_bound_is_below_the_point_estimate() -> None:
    """A lower percentile of the resampled distribution cannot exceed the full-sample z."""
    rng = np.random.default_rng(11)
    part_counts = rng.integers(0, 30, size=(24, 6)).astype(float)
    part_sizes = part_counts.sum(axis=1) + 500.0
    reference = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
    background = part_counts.sum(axis=0) + reference

    _, point = log_odds_dirichlet(
        part_counts.sum(axis=0), reference, background, target_total=part_sizes.sum()
    )
    low = bootstrap_z(
        part_counts, part_sizes, reference, background, draws=200, percentile=5.0
    )
    assert np.all(low <= point + 1e-9)


def test_bootstrap_is_reproducible() -> None:
    rng = np.random.default_rng(3)
    part_counts = rng.integers(0, 20, size=(12, 4)).astype(float)
    part_sizes = part_counts.sum(axis=1) + 100.0
    reference = np.full(4, 40.0)
    background = part_counts.sum(axis=0) + reference
    kwargs = dict(draws=50, seed=99)
    first = bootstrap_z(part_counts, part_sizes, reference, background, **kwargs)
    second = bootstrap_z(part_counts, part_sizes, reference, background, **kwargs)
    assert first == pytest.approx(second)


def test_bootstrap_needs_more_than_one_part() -> None:
    with pytest.raises(ValueError, match="two parts"):
        bootstrap_z(
            np.array([[5.0]]), np.array([10.0]), np.array([5.0]), np.array([10.0]), draws=5
        )


def test_zipf_slope_of_a_zipfian_corpus_is_near_minus_one() -> None:
    ranks = np.arange(1, 5001, dtype=float)
    counts = np.floor(1e6 / ranks)
    slope, _ = zipf_fit(counts)
    assert slope == pytest.approx(-1.0, abs=0.02)


def test_zipf_fit_refuses_a_corpus_that_is_too_small() -> None:
    with pytest.raises(ValueError, match="too few"):
        zipf_fit(np.array([5.0, 3.0, 1.0]))


def test_dispersion_excess_removes_the_frequency_confound() -> None:
    """The defect F5 fixed: a flat DP ceiling rejects rare words for being rare.

    Rare words are constructed to be *typically* concentrated and common words *typically*
    spread, mimicking the measured -0.768 correlation. A rare word sitting at its band's
    median must come out with ~zero excess even though its raw DP is high.
    """
    counts = np.concatenate([np.full(60, 25.0), np.full(60, 800.0)])
    dp = np.concatenate([np.full(60, 0.80), np.full(60, 0.20)])
    expected, excess = dispersion_excess(dp, counts, bins=2, min_per_bin=25)

    assert expected[:60] == pytest.approx(0.80)
    assert expected[60:] == pytest.approx(0.20)
    # Both groups are exactly typical for their frequency, so neither deviates.
    assert excess == pytest.approx(np.zeros_like(excess))


def test_dispersion_excess_still_flags_a_genuine_outlier() -> None:
    counts = np.concatenate([np.full(60, 25.0), np.full(60, 800.0)])
    dp = np.concatenate([np.full(60, 0.80), np.full(60, 0.20)])
    dp[0] = 0.99  # a rare word far more concentrated than other rare words
    _, excess = dispersion_excess(dp, counts, bins=2, min_per_bin=25)
    assert excess[0] > 0.15
    assert np.all(excess[1:60] <= 0.0)


def test_dispersion_excess_returns_unknown_when_it_cannot_estimate() -> None:
    """Too few words to form a band: nan, not zero. Zero would claim 'nothing deviates'."""
    expected, excess = dispersion_excess(np.array([0.5, 0.6]), np.array([10.0, 20.0]))
    assert np.all(np.isnan(expected))
    assert np.all(np.isnan(excess))


def test_dispersion_excess_ignores_absent_words() -> None:
    counts = np.concatenate([np.full(30, 25.0), np.full(30, 800.0)])
    dp = np.concatenate([np.full(30, 0.80), np.full(30, 0.20)])
    dp[5] = np.nan  # a word with no occurrences
    _, excess = dispersion_excess(dp, counts, bins=2, min_per_bin=25)
    assert np.isnan(excess[5])
    assert np.isfinite(excess[6])


def test_prior_is_not_renormalised_over_the_candidate_subset() -> None:
    """F6: a word at the same rate in both corpora must not score as over-used.

    Reproduces the real failure. Two corpora of 400k and 60M with a candidate vocabulary of
    three entries whose counts are a tiny fraction of either total. The word has an identical
    rate in both, so the honest answer is z near zero. Without the true background total the
    prior is scaled up by (all tokens / candidate tokens) and manufactures a large z.
    """
    target_total, reference_total = 400_000.0, 60_000_000.0
    rate = 75e-6
    target = np.array([target_total * rate, 20.0, 15.0])
    reference = np.array([reference_total * rate, 3000.0, 2000.0])
    background = target + reference
    background_total = target_total + reference_total

    _, honest = log_odds_dirichlet(
        target,
        reference,
        background,
        prior_mass=5000.0,
        target_total=target_total,
        reference_total=reference_total,
        background_total=background_total,
    )
    _, inflated = log_odds_dirichlet(
        target,
        reference,
        background,
        prior_mass=5000.0,
        target_total=target_total,
        reference_total=reference_total,
    )
    assert abs(honest[0]) < 1.0, "equal rates must not produce a significant z"
    assert inflated[0] > 10 * abs(honest[0]), "the bug being guarded against must be visible"

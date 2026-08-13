"""Domain separation and threshold recalibration."""

from __future__ import annotations

import numpy as np
import pytest

from zipf.domain import DOMAIN_THRESHOLD, empirical_threshold, specialisation


def tiers(**rates: tuple[float, float]) -> dict[str, tuple[dict[str, int], int]]:
    """Build tier totals from (per-million rate, per-million rate) pairs for two words."""
    size = 10_000_000
    return {
        name: (
            {"domainword": int(a * size / 1e6), "styleword": int(b * size / 1e6)},
            size,
        )
        for name, (a, b) in rates.items()
    }


def test_specialisation_is_positive_for_domain_vocabulary() -> None:
    totals = tiers(
        biomedical=(400.0, 50.0),
        technical=(400.0, 50.0),
        literature=(2.0, 50.0),
        reddit=(2.0, 50.0),
        web=(2.0, 50.0),
    )
    scores = specialisation(["domainword", "styleword"], totals)
    assert scores[0] > DOMAIN_THRESHOLD
    assert abs(scores[1]) < 0.2


def test_specialisation_ignores_the_target_corpus_entirely() -> None:
    """The score must be an external yardstick, so target counts cannot be an input.

    The function takes only reference tiers; this test pins the signature so a future change
    that starts feeding target counts in has to break it deliberately.
    """
    totals = tiers(
        biomedical=(400.0, 50.0),
        technical=(400.0, 50.0),
        literature=(2.0, 50.0),
        reddit=(2.0, 50.0),
        web=(2.0, 50.0),
    )
    first = specialisation(["domainword"], totals)
    totals_with_claude = dict(totals)
    totals_with_claude["claude_main"] = ({"domainword": 999_999}, 1_000_000)
    second = specialisation(["domainword"], totals_with_claude)
    assert first == pytest.approx(second)


def test_specialisation_is_unknown_without_both_groups() -> None:
    """nan, not zero. 'No evidence about specialisation' is not 'not specialised'."""
    only_general = tiers(literature=(5.0, 5.0), reddit=(5.0, 5.0), web=(5.0, 5.0))
    assert np.all(np.isnan(specialisation(["domainword"], only_general)))


def test_specialisation_smoothing_avoids_infinities() -> None:
    totals = tiers(
        biomedical=(300.0, 0.0),
        technical=(300.0, 0.0),
        literature=(0.0, 40.0),
        reddit=(0.0, 40.0),
        web=(0.0, 40.0),
    )
    scores = specialisation(["domainword", "styleword"], totals)
    assert np.all(np.isfinite(scores))


def test_empirical_threshold_reads_the_null_quantile() -> None:
    null = np.arange(0.0, 100.0)
    assert empirical_threshold(null, false_positive_rate=0.10) == pytest.approx(
        np.quantile(null, 0.90)
    )


def test_a_wider_null_demands_a_higher_threshold() -> None:
    """The whole point: a corpus with more topical spread must be judged more strictly."""
    narrow = np.random.default_rng(1).normal(0, 1, 20_000)
    wide = np.random.default_rng(1).normal(0, 3, 20_000)
    assert empirical_threshold(wide, false_positive_rate=0.01) > empirical_threshold(
        narrow, false_positive_rate=0.01
    )


def test_empty_null_is_refused() -> None:
    with pytest.raises(ValueError, match="null distribution is empty"):
        empirical_threshold(np.array([np.nan, np.inf]))


def test_version_control_filter_is_a_rate_ratio_not_a_significance_test() -> None:
    """F10 is why: a min-z gate on a small corpus drops words for lack of evidence.

    A rate-ratio filter has no such failure mode. A word the documentation never uses gets an
    infinite ratio, which is the correct verdict — "not explained by version control" — rather
    than a collapsed z that would read as "not over-used".
    """
    import polars as pl

    from zipf.domain import VCS_EXPLAINED_RATIO

    frame = pl.DataFrame(
        {
            "token": ["commit", "gap", "unseen"],
            "target_per_million": [3788.0, 719.0, 500.0],
            "vcs_per_million": [6383.0, 0.0, float("nan")],
        }
    ).with_columns(
        is_version_control=(
            pl.col("vcs_per_million").is_not_nan()
            & (pl.col("vcs_per_million") > 0)
            & (pl.col("target_per_million") < VCS_EXPLAINED_RATIO * pl.col("vcs_per_million"))
        )
    )
    verdict = dict(zip(frame["token"], frame["is_version_control"], strict=True))
    assert verdict["commit"] is True, "documentation uses it more than Claude does"
    assert verdict["gap"] is False, "absent from the documentation is not explained by it"
    assert verdict["unseen"] is False, "an unbuilt corpus explains nothing"

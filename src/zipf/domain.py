"""Separate domain vocabulary from style, and recalibrate what counts as a deviation.

The target corpus is bioinformatics-heavy. Words like `chromosome`, `annotation` and `variant`
are far more frequent in it than in general English, and they are not stylistic tics — they are
what the sessions were about. Left unhandled they do two separate kinds of damage:

1. **They occupy the ranking.** Every domain noun that outranks a discourse word is a slot the
   report spends on something the reader already knows.
2. **They move the yardstick.** A corpus with a heavy topical component has a wider spread of
   log-odds than one without, so a z-score of 3 does not mean the same thing here as it would
   in a topic-matched comparison. The threshold has to come from the data, not from a constant.

Two independent instruments, because they fail differently:

:func:`specialisation`
    How much more a word is used in *specialist* human writing (biomedical, technical) than in
    *general* human writing (literary, conversational, web). **Computed from the reference
    corpora only — the target is not an input.** That matters: a domain score derived partly
    from the corpus being tested would be the measurement checking itself, and every domain
    word would helpfully confirm it was a domain word. Because it is external, it can be used
    to interpret the ranking without contaminating it.
:func:`project_dispersion`
    How evenly a word is spread across the *projects* in the target corpus, rather than across
    sessions. A word used in one repository is that repository's subject; a word used in all of
    them is a habit. This catches domain terms that the reference corpora happen not to contain
    — a private vocabulary no public corpus can score.

:func:`empirical_threshold` then answers the second kind of damage directly: it reads the
z-score a word must reach to be rarer than a stated fraction of the *null* distribution, where
the null is the corpus compared against itself. That is a yardstick built from this corpus's
own topical spread rather than assumed from a normal approximation.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from zipf.counts import CorpusCounts
from zipf.harvest import DOCUMENTS_PARQUET
from zipf.pipeline import load_totals
from zipf.stats import gries_dp

logger = logging.getLogger(__name__)

#: Human corpora written by specialists about specialist subjects.
SPECIALIST_TIERS: tuple[str, ...] = ("biomedical", "technical")

#: Human corpora of general-audience English.
GENERAL_TIERS: tuple[str, ...] = ("literature", "reddit", "web")

#: Smoothing added to both sides of the specialisation ratio, in occurrences per million. Stops
#: a word absent from one side producing an infinite score.
SPECIALISATION_SMOOTHING = 0.5

#: A word this many doublings more frequent in specialist than in general writing is treated as
#: domain vocabulary. 2.0 means four times more frequent — well outside ordinary variation, and
#: chosen before looking at which words it captures.
DOMAIN_THRESHOLD = 2.0

#: The corpus of version-control documentation. Read as an instrument, never as a gate.
VCS_CORPUS = "vcs"

#: How distinctive a word must be to version-control writing before it can be dismissed as the
#: subject's vocabulary: 2.0 doublings means the git manual uses it at least four times more
#: often than general English does.
#:
#: Without this the filter is catastrophically over-broad, and quietly so. A rate ratio against
#: the manual alone flags any word the manual uses at a similar rate to Claude — which is most
#: ordinary words, because git documentation is dense technical prose. The first version of this
#: filter removed `the`, `one`, `only`, `run` and `same` from the results as "version-control
#: vocabulary".
VCS_SPECIALISATION_THRESHOLD = 2.0

#: Given that a word IS distinctive to version-control writing, it is dismissed only if Claude
#: does not out-use the manual: a Claude rate below this multiple of the documentation rate.
#:
#: This is a **rate-ratio filter, not a min-z gate**, and the distinction is the whole reason it
#: is allowed. The documentation corpus is a third of a million tokens; as a tier in the
#: minimum-z rule it would drop words whose count in it is *zero* — collapsing their z through
#: the mechanism in F10 — which looks like topical control and is really a lack of evidence. A
#: direct rate comparison has no such failure mode: a word absent from the documentation simply
#: gets an infinite ratio, which correctly reads as "not explained by version control".
VCS_EXPLAINED_RATIO = 2.0


def specialisation(
    vocabulary: list[str],
    tier_totals: dict[str, tuple[dict[str, int], int]],
) -> np.ndarray:
    """log2(specialist rate / general rate) per word, from human corpora only.

    Positive means specialist writing uses the word more than general writing does, which is
    what "domain vocabulary" means operationally. The target corpus contributes nothing.

    Returns all-``nan`` if neither group of tiers is available, because "no evidence about
    specialisation" is not "not specialised".
    """
    def pooled(names: tuple[str, ...]) -> tuple[np.ndarray, float]:
        counts = np.zeros(len(vocabulary), dtype=np.float64)
        size = 0.0
        for name in names:
            if name not in tier_totals:
                continue
            totals, tier_size = tier_totals[name]
            counts += np.array([totals.get(t, 0) for t in vocabulary], dtype=np.float64)
            size += float(tier_size)
        return counts, size

    specialist_counts, specialist_size = pooled(SPECIALIST_TIERS)
    general_counts, general_size = pooled(GENERAL_TIERS)
    if specialist_size <= 0 or general_size <= 0:
        logger.warning(
            "specialisation needs both a specialist and a general tier; returning unknown"
        )
        return np.full(len(vocabulary), np.nan)

    specialist_rate = specialist_counts / specialist_size * 1e6 + SPECIALISATION_SMOOTHING
    general_rate = general_counts / general_size * 1e6 + SPECIALISATION_SMOOTHING
    return np.log2(specialist_rate / general_rate)


def project_counts(
    vocabulary: list[str], *, stratum: str = "claude_main"
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-project count matrix for the target corpus, and each project's token total.

    Projects are the coarser dispersion unit. Sessions inside one project share a subject, so
    session dispersion cannot see a word that is common to a whole repository — which is
    exactly the shape a domain term has.
    """
    frame = pl.read_parquet(DOCUMENTS_PARQUET).filter(pl.col("corpus_id") == stratum)
    counts = CorpusCounts(stratum, focus_vocab=frozenset(vocabulary))
    for project, text in zip(frame["project"], frame["text"], strict=True):
        counts.add(str(project), str(text), preprocessor="markdown")

    projects = sorted(counts.part_sizes)
    index = {t: i for i, t in enumerate(vocabulary)}
    matrix = np.zeros((len(projects), len(vocabulary)), dtype=np.float64)
    for row, project in enumerate(projects):
        for token, value in counts.per_part.get(project, {}).items():
            column = index.get(token)
            if column is not None:
                matrix[row, column] = value
    sizes = np.array([counts.part_sizes[p] for p in projects], dtype=np.float64)
    return matrix, sizes, projects


def project_dispersion(vocabulary: list[str], *, stratum: str = "claude_main") -> np.ndarray:
    """Gries DP over projects. 0 = used everywhere, 1 = confined to one repository."""
    matrix, sizes, projects = project_counts(vocabulary, stratum=stratum)
    logger.info("project dispersion over %d projects", len(projects))
    return gries_dp(matrix, sizes)


def empirical_threshold(null_z: np.ndarray, *, false_positive_rate: float = 0.01) -> float:
    """The z a word must clear to beat a stated fraction of the null distribution.

    ``null_z`` is the z-scores from comparing the corpus against itself, where every value is by
    construction a false positive. Reading the threshold off that distribution replaces the
    assumed constant with one calibrated to this corpus's own topical spread.
    """
    finite = null_z[np.isfinite(null_z)]
    if finite.size == 0:
        raise ValueError("null distribution is empty; cannot calibrate a threshold")
    return float(np.quantile(finite, 1.0 - false_positive_rate))


def vcs_specialisation(
    vocabulary: list[str],
    rates: np.ndarray,
    tier_totals: dict[str, tuple[dict[str, int], int]],
) -> np.ndarray:
    """log2(rate in the git manual / rate in general English).

    High means the word belongs to writing *about version control*, which is the property that
    licenses dismissing it. A raw comparison against the manual cannot distinguish that from
    "both are ordinary prose".
    """
    general_counts = np.zeros(len(vocabulary), dtype=np.float64)
    general_size = 0.0
    for name in GENERAL_TIERS:
        if name not in tier_totals:
            continue
        totals, size = tier_totals[name]
        general_counts += np.array([totals.get(t, 0) for t in vocabulary], dtype=np.float64)
        general_size += float(size)
    if general_size <= 0:
        return np.full(len(vocabulary), np.nan)
    general_rate = general_counts / general_size * 1e6 + SPECIALISATION_SMOOTHING
    return np.log2((rates + SPECIALISATION_SMOOTHING) / general_rate)


def vcs_rates(vocabulary: list[str]) -> np.ndarray:
    """Each word's rate per million in version-control documentation.

    Returns all-``nan`` when the corpus has not been built, because "we did not look" is not
    "the documentation does not use it".
    """
    try:
        totals, size = load_totals(VCS_CORPUS)
    except FileNotFoundError:
        logger.warning(
            "the %s corpus is not built, so no word can be checked against version-control "
            "documentation; run scripts/fetch_vcs_corpus.py then `zipf count --corpus vcs`",
            VCS_CORPUS,
        )
        return np.full(len(vocabulary), np.nan)
    return np.array([totals.get(t, 0) for t in vocabulary], dtype=np.float64) / size * 1e6


def annotate(
    wide: pl.DataFrame,
    tier_totals: dict[str, tuple[dict[str, int], int]],
    *,
    stratum: str = "claude_main",
) -> pl.DataFrame:
    """Add specialisation, project dispersion and the version-control rate."""
    vocabulary = wide["token"].to_list()
    scores = specialisation(vocabulary, tier_totals)
    dispersion = project_dispersion(vocabulary, stratum=stratum)
    vcs = vcs_rates(vocabulary)
    vcs_spec = vcs_specialisation(vocabulary, vcs, tier_totals)
    return wide.with_columns(
        specialisation=pl.Series(scores),
        project_dp=pl.Series(dispersion),
        vcs_per_million=pl.Series(vcs),
        vcs_specialisation=pl.Series(vcs_spec),
    ).with_columns(
        is_domain=(pl.col("specialisation") >= DOMAIN_THRESHOLD),
        # Two conditions, and both are needed. Distinctive to version-control writing, AND not
        # out-used by Claude. Dropping the first condition removes ordinary words; dropping the
        # second would dismiss a word the manual mentions once and Claude says constantly.
        is_version_control=(
            pl.col("vcs_per_million").is_not_nan()
            & (pl.col("vcs_per_million") > 0)
            & (pl.col("vcs_specialisation") >= VCS_SPECIALISATION_THRESHOLD)
            & (pl.col("target_per_million") < VCS_EXPLAINED_RATIO * pl.col("vcs_per_million"))
        ),
    )

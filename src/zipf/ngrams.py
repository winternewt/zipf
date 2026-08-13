"""Phrase-level extension: 2-, 3- and 4-gram chains.

Gated deliberately. The user's instruction was to establish the unigram signal first and only
then extend, because this stage costs another full pass over every corpus and is worthless if
the single-word comparison shows nothing.

**Why this cannot reuse the unigram counter.** A 60-million-token corpus yields roughly as
many n-grams as tokens, but far more *types*: unigram types plateau in the hundreds of
thousands while 4-gram types run into the tens of millions. Holding four such tables for four
tiers does not fit in memory and is not needed, because only n-grams that occur in the target
corpus can ever be reported. So the target is counted exhaustively, its frequent n-grams
become a **candidate set**, and each reference tier counts only members of that set while
tracking its true n-gram total separately for the denominator.

That asymmetry is safe in one direction and unsafe in the other. Restricting which n-grams a
reference corpus *records* does not bias any rate, because the denominator is counted over the
full stream. Restricting which n-grams the *target* records would, which is why the target
side is exhaustive.

Overlapping n-grams are not independent observations — `a b c` contributes to two bigrams that
share a token — so the z-scores here are optimistic relative to the unigram ones. They order
phrases correctly; they should not be read as calibrated significance.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Iterator

import numpy as np
import polars as pl

from zipf.corpora import TIERS, TierPlan
from zipf.corpora import iter_documents as iter_tier_documents
from zipf.counts import Document
from zipf.harvest import DOCUMENTS_PARQUET
from zipf.models import REFERENCE_TIERS
from zipf.paths import OUTPUT_DIR
from zipf.stats import DEFAULT_PRIOR_MASS, gries_dp, log_likelihood_g2, log_odds_dirichlet, occupancy
from zipf.tokenize import iter_tokens

logger = logging.getLogger(__name__)

#: Chain lengths to build.
NGRAM_SIZES: tuple[int, ...] = (2, 3, 4)

#: Minimum occurrences in the target corpus for an n-gram to become a candidate. Longer chains
#: are rarer, so the floor drops with length; below these the prior dominates entirely.
MIN_TARGET_COUNT: dict[int, int] = {2: 20, 3: 10, 4: 8}

#: Cap on reference tokens read per tier, mirroring the unigram build so the two stages compare
#: like with like.
DEFAULT_TOKEN_CAP = 60_000_000

SEPARATOR = " "


def iter_ngrams(tokens: Iterable[str], n: int) -> Iterator[str]:
    """Yield space-joined n-grams from a token stream.

    N-grams do not cross document boundaries, because the caller passes one document's tokens
    at a time. A chain spanning two unrelated replies would be an artifact of concatenation.
    """
    window: list[str] = []
    for token in tokens:
        window.append(token)
        if len(window) > n:
            window.pop(0)
        if len(window) == n:
            yield SEPARATOR.join(window)


class NgramCounts:
    """Counts for one corpus at one chain length.

    ``totals`` may be restricted to a candidate set, but ``total`` — the denominator — is
    always the full number of n-grams seen.
    """

    def __init__(self, corpus_id: str, n: int, *, candidates: frozenset[str] | None = None) -> None:
        self.corpus_id = corpus_id
        self.n = n
        self.candidates = candidates
        self.totals: Counter[str] = Counter()
        self.per_part: dict[str, Counter[str]] = {}
        self.part_sizes: Counter[str] = Counter()
        self.total = 0

    def add_tokens(self, part_id: str, tokens: list[str], *, track_parts: bool) -> None:
        grams = list(iter_ngrams(tokens, self.n))
        if not grams:
            return
        self.total += len(grams)
        self.part_sizes[part_id] += len(grams)
        if self.candidates is None:
            local = Counter(grams)
        else:
            local = Counter(g for g in grams if g in self.candidates)
            if not local:
                return
        self.totals.update(local)
        if track_parts:
            self.per_part.setdefault(part_id, Counter()).update(local)


def count_target(
    stratum: str = "claude_main", sizes: tuple[int, ...] = NGRAM_SIZES
) -> dict[int, NgramCounts]:
    """Count every n-gram in the target corpus, exhaustively."""
    frame = pl.read_parquet(DOCUMENTS_PARQUET).filter(pl.col("corpus_id") == stratum)
    counters = {n: NgramCounts(stratum, n) for n in sizes}
    for part_id, text in zip(frame["part_id"], frame["text"], strict=True):
        tokens = list(iter_tokens(str(text), preprocessor="markdown"))
        for counter in counters.values():
            counter.add_tokens(str(part_id), tokens, track_parts=True)
    for n, counter in counters.items():
        logger.info("%s %d-grams: %d total, %d types", stratum, n, counter.total, len(counter.totals))
    return counters


def candidate_sets(counters: dict[int, NgramCounts]) -> dict[int, frozenset[str]]:
    """The n-grams frequent enough in the target to be worth measuring anywhere else."""
    sets = {}
    for n, counter in counters.items():
        floor = MIN_TARGET_COUNT[n]
        sets[n] = frozenset(g for g, c in counter.totals.items() if c >= floor)
        logger.info("%d-gram candidates at count >= %d: %d", n, floor, len(sets[n]))
    return sets


def count_reference(
    tier: TierPlan,
    candidates: dict[int, frozenset[str]],
    *,
    token_cap: int = DEFAULT_TOKEN_CAP,
    sizes: tuple[int, ...] = NGRAM_SIZES,
) -> dict[int, NgramCounts]:
    """Count candidate n-grams in one reference tier, in a single tokenizing pass."""
    files = [tier.directory / name for name in tier.files]
    missing = [p for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"tier {tier.corpus_id!r} is missing {missing[0]}; run `uv run zipf fetch` first"
        )
    counters = {n: NgramCounts(tier.corpus_id, n, candidates=candidates[n]) for n in sizes}
    seen_tokens = 0
    documents: Iterable[Document] = iter_tier_documents(tier, files)
    for part_id, text in documents:
        if seen_tokens >= token_cap:
            logger.warning(
                "%s: stopped at the %d-token cap; the tier holds more text than was read",
                tier.corpus_id,
                token_cap,
            )
            break
        tokens = list(iter_tokens(text, preprocessor=tier.preprocessor))
        seen_tokens += len(tokens)
        for counter in counters.values():
            counter.add_tokens(part_id, tokens, track_parts=False)
    for n, counter in counters.items():
        logger.info("%s %d-grams: %d total, %d candidate types seen", tier.corpus_id, n, counter.total, len(counter.totals))
    return counters


def compare_ngrams(
    n: int,
    target: NgramCounts,
    references: dict[str, NgramCounts],
    *,
    prior_mass: float = DEFAULT_PRIOR_MASS,
) -> pl.DataFrame:
    """Rank one chain length against every reference tier, minimum-z across tiers."""
    floor = MIN_TARGET_COUNT[n]
    vocabulary = sorted(g for g, c in target.totals.items() if c >= floor)
    if not vocabulary:
        raise ValueError(f"no {n}-gram reaches the minimum count of {floor}")

    target_counts = np.array([target.totals[g] for g in vocabulary], dtype=np.float64)
    background = target_counts.copy()
    # Full n-gram totals, not the sum over candidates. With only a few hundred candidate
    # 4-grams the subset-normalised prior dominated the real counts entirely.
    background_total = float(target.total)
    for counts in references.values():
        background += np.array([counts.totals.get(g, 0) for g in vocabulary], dtype=np.float64)
        background_total += float(counts.total)

    part_ids = sorted(target.per_part)
    index = {g: i for i, g in enumerate(vocabulary)}
    matrix = np.zeros((len(part_ids), len(vocabulary)), dtype=np.float64)
    for row, part_id in enumerate(part_ids):
        for gram, value in target.per_part[part_id].items():
            column = index.get(gram)
            if column is not None:
                matrix[row, column] = value
    sizes = np.array([target.part_sizes[p] for p in part_ids], dtype=np.float64)

    dispersion = gries_dp(matrix, sizes)
    present, share = occupancy(matrix)

    frames = []
    for name, counts in references.items():
        reference_counts = np.array(
            [counts.totals.get(g, 0) for g in vocabulary], dtype=np.float64
        )
        _, z = log_odds_dirichlet(
            target_counts,
            reference_counts,
            background,
            prior_mass=prior_mass,
            target_total=target.total,
            reference_total=counts.total,
            background_total=background_total,
        )
        g2 = log_likelihood_g2(
            target_counts,
            reference_counts,
            target_total=target.total,
            reference_total=counts.total,
        )
        frames.append(
            pl.DataFrame(
                {
                    "ngram": vocabulary,
                    "reference": [name] * len(vocabulary),
                    "z": z,
                    "g2": g2,
                    "reference_count": reference_counts.astype(np.int64),
                    "reference_per_million": reference_counts / counts.total * 1e6,
                }
            )
        )

    long = pl.concat(frames).with_columns(agrees=(pl.col("z") >= 3.0) & (pl.col("g2") > 0))
    wide = (
        long.group_by("ngram")
        .agg(
            pl.col("z").min().alias("z_min"),
            pl.col("agrees").sum().alias("tiers_agreeing"),
            pl.len().alias("tiers_compared"),
            pl.col("reference_per_million").max().alias("best_reference_per_million"),
        )
        .join(
            pl.DataFrame(
                {
                    "ngram": vocabulary,
                    "target_count": target_counts.astype(np.int64),
                    "target_per_million": target_counts / target.total * 1e6,
                    "dispersion_dp": dispersion,
                    "sessions_present": present.astype(np.int64),
                    "max_session_share": share,
                }
            ),
            on="ngram",
            how="left",
        )
        .with_columns(n=pl.lit(n, dtype=pl.Int8))
        .sort(["z_min", "ngram"], descending=[True, False])
    )
    return wide


def run(
    *,
    sizes: tuple[int, ...] = NGRAM_SIZES,
    tiers: tuple[str, ...] = REFERENCE_TIERS,
    token_cap: int = DEFAULT_TOKEN_CAP,
) -> dict[int, pl.DataFrame]:
    """Full n-gram stage: count target, count tiers, compare, persist."""
    target = count_target(sizes=sizes)
    candidates = candidate_sets(target)

    references: dict[str, dict[int, NgramCounts]] = {}
    for name in tiers:
        references[name] = count_reference(
            TIERS[name], candidates, token_cap=token_cap, sizes=sizes
        )

    results: dict[int, pl.DataFrame] = {}
    for n in sizes:
        frame = compare_ngrams(n, target[n], {k: v[n] for k, v in references.items()})
        frame.write_parquet(OUTPUT_DIR / f"overuse_ngram_{n}.parquet")
        results[n] = frame
        logger.info("%d-grams: wrote %d rows", n, frame.height)
    return results

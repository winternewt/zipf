"""Turn documents into frequency tables.

Two tables come out of a corpus and they answer different questions:

``totals``
    ``token -> count`` over the whole corpus, plus the corpus token total. This is the
    denominator every rate is computed against, and it is always computed over the *full*
    stream — never over a filtered vocabulary — or the rates would be wrong.
``per_part``
    ``(part_id, token) -> count``. Only this table supports dispersion, which is what stops a
    single long session from looking like a stylistic tic.

For a corpus in the hundreds of millions of tokens the per-part table is the expensive one, so
it may be restricted to a *focus vocabulary*. Restricting which tokens get a per-part
breakdown does not bias any rate, because the totals table is unfiltered.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Iterator

import polars as pl

from zipf.tokenize import iter_tokens

logger = logging.getLogger(__name__)

#: A document as it enters counting: which dispersion part it belongs to, and its raw text.
Document = tuple[str, str]


class CorpusCounts:
    """Accumulated counts for one corpus.

    Mutable by design: a corpus arrives as a stream of shards and this is folded over them.
    """

    def __init__(self, corpus_id: str, *, focus_vocab: frozenset[str] | None = None) -> None:
        self.corpus_id = corpus_id
        self.focus_vocab = focus_vocab
        self.totals: Counter[str] = Counter()
        self.per_part: dict[str, Counter[str]] = {}
        self.part_sizes: Counter[str] = Counter()
        self.documents = 0
        self.tokens = 0

    def add(self, part_id: str, text: str, *, preprocessor: str) -> None:
        """Fold one document in."""
        local: Counter[str] = Counter(iter_tokens(text, preprocessor=preprocessor))
        if not local:
            # An empty document is legitimate — a reply that was entirely a code block leaves
            # no prose. It counts as a document, contributes no tokens, and is not an error.
            self.documents += 1
            return
        total = sum(local.values())
        self.totals.update(local)
        self.documents += 1
        self.tokens += total
        self.part_sizes[part_id] += total
        if self.focus_vocab is None:
            tracked = local
        else:
            tracked = Counter({t: c for t, c in local.items() if t in self.focus_vocab})
            if not tracked:
                return
        bucket = self.per_part.setdefault(part_id, Counter())
        bucket.update(tracked)

    def extend(self, documents: Iterable[Document], *, preprocessor: str) -> None:
        for part_id, text in documents:
            self.add(part_id, text, preprocessor=preprocessor)

    @property
    def types(self) -> int:
        return len(self.totals)

    @property
    def parts(self) -> int:
        return len(self.part_sizes)

    def totals_frame(self) -> pl.DataFrame:
        """``token, count`` sorted by ``(-count, token)`` so ties are stable across runs."""
        if not self.totals:
            return pl.DataFrame(schema={"token": pl.Utf8, "count": pl.UInt64})
        frame = pl.DataFrame(
            {
                "token": list(self.totals.keys()),
                "count": list(self.totals.values()),
            },
            schema={"token": pl.Utf8, "count": pl.UInt64},
        )
        return frame.sort(["count", "token"], descending=[True, False])

    def per_part_frame(self) -> pl.DataFrame:
        """``part_id, token, count`` sorted deterministically."""
        parts: list[str] = []
        tokens: list[str] = []
        counts: list[int] = []
        for part_id in sorted(self.per_part):
            bucket = self.per_part[part_id]
            for token in sorted(bucket):
                parts.append(part_id)
                tokens.append(token)
                counts.append(bucket[token])
        return pl.DataFrame(
            {"part_id": parts, "token": tokens, "count": counts},
            schema={"part_id": pl.Utf8, "token": pl.Utf8, "count": pl.UInt64},
        )

    def part_sizes_frame(self) -> pl.DataFrame:
        """``part_id, size`` — the per-part denominators dispersion needs."""
        items = sorted(self.part_sizes.items())
        return pl.DataFrame(
            {"part_id": [k for k, _ in items], "size": [v for _, v in items]},
            schema={"part_id": pl.Utf8, "size": pl.UInt64},
        )


def documents_from_frame(frame: pl.DataFrame, *, part_column: str = "part_id") -> Iterator[Document]:
    """Stream ``(part_id, text)`` pairs out of a harvested documents table."""
    for part_id, text in zip(frame[part_column], frame["text"], strict=True):
        yield str(part_id), str(text)


def count_corpus(
    corpus_id: str,
    documents: Iterable[Document],
    *,
    preprocessor: str,
    focus_vocab: frozenset[str] | None = None,
) -> CorpusCounts:
    """Count a whole corpus in one pass."""
    counts = CorpusCounts(corpus_id, focus_vocab=focus_vocab)
    counts.extend(documents, preprocessor=preprocessor)
    logger.info(
        "%s: %d documents, %d parts, %d tokens, %d types",
        corpus_id,
        counts.documents,
        counts.parts,
        counts.tokens,
        counts.types,
    )
    return counts

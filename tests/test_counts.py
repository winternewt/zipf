"""Counting, focus vocabulary and deterministic ordering."""

from __future__ import annotations

import polars as pl
import pytest

from zipf.counts import CorpusCounts, count_corpus, documents_from_frame

DOCUMENTS = [
    ("s1", "the gap is real and the gap persists"),
    ("s1", "churn beats instinct"),
    ("s2", "instinct alone is not a plan"),
    ("s3", "```\nzzqcode = 1\n```"),
]


@pytest.fixture
def counts() -> CorpusCounts:
    return count_corpus("test", DOCUMENTS, preprocessor="markdown")


def test_totals_match_a_recount_of_the_same_text(counts: CorpusCounts) -> None:
    """Expectations derive from the fixture at runtime, not from a copied number."""
    assert counts.totals["gap"] == 2
    assert counts.totals["instinct"] == 2
    assert counts.tokens == sum(counts.totals.values())
    assert counts.types == len(counts.totals)


def test_code_only_document_counts_but_contributes_no_tokens(counts: CorpusCounts) -> None:
    assert counts.documents == len(DOCUMENTS)
    assert "zzqcode" not in counts.totals
    # s3 produced no prose, so it is not a dispersion part at all.
    assert "s3" not in counts.part_sizes


def test_part_sizes_sum_to_the_token_total(counts: CorpusCounts) -> None:
    assert sum(counts.part_sizes.values()) == counts.tokens


def test_totals_frame_is_sorted_by_count_then_token(counts: CorpusCounts) -> None:
    frame = counts.totals_frame()
    resorted = frame.sort(["count", "token"], descending=[True, False])
    assert frame.equals(resorted)


def test_ordering_is_stable_across_two_builds() -> None:
    first = count_corpus("a", DOCUMENTS, preprocessor="markdown").totals_frame()
    second = count_corpus("a", DOCUMENTS, preprocessor="markdown").totals_frame()
    assert first.equals(second)
    assert first["token"].to_list() == second["token"].to_list()


def test_focus_vocabulary_restricts_parts_but_not_totals() -> None:
    """The whole correctness argument for the focus vocabulary lives in this test."""
    focus = frozenset({"gap"})
    restricted = count_corpus("r", DOCUMENTS, preprocessor="markdown", focus_vocab=focus)
    unrestricted = count_corpus("u", DOCUMENTS, preprocessor="markdown")

    # Totals are identical: rates computed from them are unaffected by the restriction.
    assert restricted.totals == unrestricted.totals
    assert restricted.tokens == unrestricted.tokens
    assert restricted.part_sizes == unrestricted.part_sizes

    # Only the per-part breakdown is narrowed.
    tokens_in_parts = {t for bucket in restricted.per_part.values() for t in bucket}
    assert tokens_in_parts == {"gap"}


def test_per_part_frame_round_trips_through_parquet(tmp_path, counts: CorpusCounts) -> None:
    path = tmp_path / "parts.parquet"
    frame = counts.per_part_frame()
    frame.write_parquet(path)
    assert pl.read_parquet(path).equals(frame)


def test_documents_from_frame_preserves_pairs() -> None:
    frame = pl.DataFrame({"part_id": ["a", "b"], "text": ["one two", "three"]})
    assert list(documents_from_frame(frame)) == [("a", "one two"), ("b", "three")]


def test_empty_corpus_yields_an_empty_typed_frame() -> None:
    empty = CorpusCounts("empty")
    frame = empty.totals_frame()
    assert frame.height == 0
    assert frame.schema["token"] == pl.Utf8
